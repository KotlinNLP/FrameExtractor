/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers

import com.kotlinnlp.frameextractor.helpers.dataset.IOBTag
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.frameextractor.FramesExtractor
import com.kotlinnlp.frameextractor.FramesExtractorModel
import com.kotlinnlp.frameextractor.TextFramesExtractorModel
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.utils.ShuffledIndices
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * A helper to train a [FramesExtractorModel].
 *
 * @param model the model to train
 * @param modelFilename the path of the file in which to save the serialized trained model
 * @param epochs the number of training epochs
 * @param extractorUpdateMethod the update method to optimize the frame extractor model parameters
 * @param encoderUpdateMethod the update method for the parameters of the tokens encoder (null if must not be trained)
 * @param useDropout whether to apply the dropout of the input
 * @param validator a helper for the validation of the model
 * @param verbose whether to print info about the training progress and timing (default = true)
 */
class Trainer(
  private val model: TextFramesExtractorModel,
  private val modelFilename: String,
  private val epochs: Int,
  extractorUpdateMethod: UpdateMethod<*> = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  encoderUpdateMethod: UpdateMethod<*>? = null,
  useDropout: Boolean,
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
  private val timer = Timer()

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = -1.0 // -1 used as init value (all accuracy values are in the range [0.0, 1.0])

  /**
   * A frame extractor built with the given [model].
   */
  private val extractor = FramesExtractor(
    model = this.model.frameExtractor,
    propagateToInput = encoderUpdateMethod != null)

  /**
   * The encoder of the input tokens.
   */
  private val encoder: TokensEncoder<FormToken, Sentence<FormToken>> =
    this.model.tokensEncoder.buildEncoder(useDropout)

  /**
   * The optimizer of the [model] parameters.
   */
  private val extractorOptimizer = ParamsOptimizer(extractorUpdateMethod)

  /**
   * The optimizer of the tokens encoder.
   * It is null if it must not be trained.
   */
  private val encoderOptimizer: ParamsOptimizer? = encoderUpdateMethod?.let { ParamsOptimizer(it) }

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
  fun train(dataset: Dataset, shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743)) {

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch(dataset = dataset, shuffler = shuffler)

      this.logTrainingEnd()

      this.validateAndSaveModel()
    }
  }

  /**
   * Train the [model] with all the examples of the dataset.
   *
   * @param dataset the training dataset
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  private fun trainEpoch(dataset: Dataset, shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(dataset.examples.size)

    ShuffledIndices(dataset.examples.size, shuffler = shuffler).forEach { i ->

      val example: Dataset.Example = dataset.examples[i]
      val intentIndex: Int = dataset.configuration.indexOfFirst { it.name == example.intent }

      if (this.verbose) progress.tick()

      this.newBatch()

      this.trainExample(
        example = example,
        intentIndex = intentIndex,
        intentConfig = dataset.configuration[intentIndex],
        slotsOffset = this.model.frameExtractor.getSlotsOffset(example.intent))

      // TODO: setup newExample() call?

      this.update() // the params errors copies are optimized considering batches with a single example
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
  private fun trainExample(example: Dataset.Example,
                           intentIndex: Int,
                           intentConfig: Intent.Configuration,
                           slotsOffset: Int) {

    val tokensEncodings: List<DenseNDArray> = this.encoder.forward(example.sentence)
    val output: FramesExtractor.Output = this.extractor.forward(tokensEncodings)

    val intentErrors: DenseNDArray = output.intentsDistribution.sub(
      DenseNDArrayFactory.oneHotEncoder(length = output.intentsDistribution.length, oneAt = intentIndex))

    val slotsErrors: List<DenseNDArray> = output.slotsClassifications.zip(example.sentence.tokens).map {
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
    this.extractorOptimizer.accumulate(this.extractor.getParamsErrors(copy = false), copy = false) // optimized copy

    this.encoderOptimizer?.let { optimizer ->
      this.encoder.backward(this.extractor.getInputErrors(copy = false))
      optimizer.accumulate(this.encoder.getParamsErrors(copy = false), copy = false) // optimized copy
    }
  }

  /**
   * Optimizers update.
   */
  private fun update() {

    this.extractorOptimizer.update()
    this.encoderOptimizer?.update()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  private fun newBatch() {

    this.extractorOptimizer.newBatch()
    this.encoderOptimizer?.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  private fun newEpoch() {

    this.extractorOptimizer.newEpoch()
    this.encoderOptimizer?.newEpoch()

    this.epochCount++
  }

  /**
   * Validate the [model] and save it to file if a new best accuracy has been reached.
   */
  private fun validateAndSaveModel() {

    this.logValidationStart()

    val stats: Statistics = this.validator.evaluate()

    this.logValidationEnd()

    println("\nStatistics\n$stats")

    if (stats.accuracy > this.bestAccuracy) {

      this.model.dump(FileOutputStream(File(this.modelFilename)))
      println("\nNEW BEST ACCURACY! Model saved to \"${this.modelFilename}\"")

      this.bestAccuracy = stats.accuracy
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
