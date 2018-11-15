/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers

import com.kotlinnlp.frameextractor.helpers.dataset.IOBTag
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.frameextractor.FrameExtractor
import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.frameextractor.helpers.dataset.EncodedDataset
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.utils.ExamplesIndices
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper to train a [FrameExtractorModel].
 *
 * @param model the model to train
 * @param modelFilename the path of the file in which to save the serialized trained model
 * @param epochs the number of training epochs
 * @param updateMethod the update method to optimize the model parameters
 * @param validator a helper for the validation of the model
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class Trainer(
  private val model: FrameExtractorModel,
  private val modelFilename: String,
  private val epochs: Int,
  private val updateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  private val validator: Validator,
  private val verbose: Boolean = true
) {

  /**
   * The epoch counter.
   */
  private var epochCount: Int = 0

  /**
   * A timer to track the elapsed time.
   */
  private var timer = Timer()

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = -1.0 // -1 used as init value (all accuracy values are in the range [0.0, 1.0])

  /**
   * A frame extractor built with the given [model].
   */
  private val extractor = FrameExtractor(this.model)

  /**
   * The optimizer of the [model] parameters.
   */
  private val optimizer = ParamsOptimizer(params = this.model.params, updateMethod = this.updateMethod)

  /**
   * Check requirements.
   */
  init {
    require(this.epochs > 0) { "The number of epochs must be > 0" }
  }

  /**
   * Train the [model] with the given training dataset over more epochs.
   *
   * @param dataset the training dataset
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  fun train(dataset: EncodedDataset, shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743)) {

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch(dataset = dataset, shuffler = shuffler)

      this.logTrainingEnd()

      this.logValidationStart()
      this.validateAndSaveModel()
      this.logValidationEnd()
    }
  }

  /**
   * Train the [model] with all the examples of the dataset.
   *
   * @param dataset the training dataset
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  private fun trainEpoch(dataset: EncodedDataset, shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(dataset.examples.size)

    ExamplesIndices(dataset.examples.size, shuffler = shuffler).forEach { i ->

      val example: EncodedDataset.Example = dataset.examples[i]
      val intentIndex: Int = dataset.configuration.indexOfFirst { it.name == example.intent }

      if (this.verbose) progress.tick()

      this.newBatch()

      this.trainExample(
        example = example,
        intentIndex = intentIndex,
        intentConfig = dataset.configuration[intentIndex],
        slotsOffset = this.extractor.getSlotsOffset(example.intent))
    }
  }

  /**
   * Train the [model] with a single example.
   *
   * @param example a training example
   * @param intentIndex the index of the intent of this example, within the dataset configuration
   * @param intentConfig the configuration of the intent of this example
   * @param slotsOffset the offset index from which the slots of a this example intent start, within the concatenation
   *                    of all the possible intents slots
   */
  private fun trainExample(example: EncodedDataset.Example,
                           intentIndex: Int,
                           intentConfig: Intent.Configuration,
                           slotsOffset: Int) {

    val output: FrameExtractor.Output = this.extractor.forward(example.tokens.map { it.encoding })

    val intentErrors: DenseNDArray = output.intentsDistribution.sub(
      DenseNDArrayFactory.oneHotEncoder(length = output.intentsDistribution.length, oneAt = intentIndex))

    val slotsErrors: List<DenseNDArray> = output.slotsClassifications.zip(example.tokens).map {
      (classification, exampleToken) ->

      val slot: Dataset.Example.Slot = exampleToken.slot
      val slotId: Int = slotsOffset + intentConfig.getSlotIndex(slot.name)
      val goldClassification: DenseNDArray = DenseNDArrayFactory.oneHotEncoder(
        length = classification.length,
        oneAt = 2 * slotId + (if (slot.iob == IOBTag.Beginning) 0 else 1))

      classification.sub(goldClassification)
    }

    this.extractor.backward(
      outputErrors = this.extractor.Output(intentsDistribution = intentErrors, slotsClassifications = slotsErrors))

    this.optimizer.accumulate(this.extractor.getParamsErrors(copy = false), copy = false)
    this.optimizer.update()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  private fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  private fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }

    this.epochCount++
  }

  /**
   * Validate the [model] and save it to file if a new best accuracy has been reached.
   */
  private fun validateAndSaveModel() {

    val stats: Statistics = this.validator.evaluate()
    val accuracy: Double = stats.intents.f1Score * stats.slots.f1Score

    println("\nStatistics\n$stats")

    if (accuracy > this.bestAccuracy) {

      this.model.dump(FileOutputStream(File(this.modelFilename)))
      println("\nNEW BEST ACCURACY! Model saved to \"${this.modelFilename}\"")

      this.bestAccuracy = accuracy
    }
  }

  /**
   * Log when training starts.
   *
   * @param epochIndex the current epoch index
   */
  private fun logTrainingStart(epochIndex: Int) {

    if (this.verbose) {

      this.timer.reset()

      println("\nEpoch ${epochIndex + 1} of ${this.epochs}")
      println("\nStart training...")
    }
  }

  /**
   * Log when training ends.
   */
  private fun logTrainingEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }

  /**
   * Log when validation starts.
   */
  private fun logValidationStart() {

    if (this.verbose) {

      this.timer.reset()

      println("\nStart validation...")
    }
  }

  /**
   * Log when validation ends.
   */
  private fun logValidationEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }
}
