/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers

import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.frameextractor.FrameExtractor
import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.dataset.EncodedDataset
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper to evaluate a [FrameExtractorModel].
 *
 * @param model the model to evaluate
 * @param dataset the validation dataset
 * @param verbose whether to print info about the validation progress (default = true)
 */
class Validator(
  private val model: FrameExtractorModel,
  private val dataset: EncodedDataset,
  private val verbose: Boolean = true
) {

  /**
   * A frame extractor built with the given [model].
   */
  private val extractor  = FrameExtractor(this.model)

  /**
   * The total number of intents in the validation dataset.
   */
  private val totalIntents: Int = this.dataset.examples.size

  /**
   * The total number of slots in the validation dataset.
   */
  private val totalSlots: Int = this.dataset.examples.sumBy { it.tokens.size }

  /**
   * The count of correct intents found.
   */
  private var correctIntents = 0

  /**
   * The count of correct slots found.
   */
  private var correctSlots = 0

  /**
   * The count of not-null slots found.
   */
  private var notNullSlots = 0

  /**
   * Evaluate the [model] and get statistics.
   *
   * @return the evaluation statistics
   */
  fun evaluate(): Statistics {

    val progress = ProgressIndicatorBar(this.dataset.examples.size)

    this.reset()

    this.dataset.examples.forEach {

      this.validateExample(it)

      if (this.verbose) progress.tick()
    }

    return this.buildStats()
  }

  /**
   * Reset the counters.
   */
  private fun reset() {

    this.correctIntents = 0
    this.correctSlots = 0
    this.notNullSlots = 0
  }

  /**
   * Validate the [model] on a single example.
   *
   * @param example an example of the dataset
   */
  private fun validateExample(example: EncodedDataset.Example) {

    val output: FrameExtractor.Output = this.extractor.forward(example.tokens.map { it.encoding })

    val bestIntentIndex: Int = output.intentsDistribution.argMaxIndex()
    val intentConfig: Intent.Configuration = this.model.intentsConfiguration[bestIntentIndex]
    val bestIntentName: String = intentConfig.name

    if (example.intent == bestIntentName) {

      this.correctIntents++

      this.validateSlots(
        example = example,
        possibleSlots = intentConfig.slots,
        slotsClassifications = output.slotsClassifications)
    }
  }

  /**
   * Validate the slots classified.
   *
   * @param example the example from which the slots have been classified
   * @param possibleSlots the list of possible slot names that can be associated to the example intent
   * @param slotsClassifications the list of slots classifications
   */
  private fun validateSlots(example: EncodedDataset.Example,
                            possibleSlots: List<String>,
                            slotsClassifications: List<DenseNDArray>) {

    val intentSlotsOffset: Int = this.extractor.getSlotsOffset(example.intent)
    val intentSlotsRange = intentSlotsOffset until (intentSlotsOffset + possibleSlots.size)

    val predictedSlotsNames: List<String?> = slotsClassifications.map {

      val bestSlotIndex: Int = it.argMaxIndex() / 2

      if (bestSlotIndex in intentSlotsRange) possibleSlots[bestSlotIndex - intentSlotsOffset] else null
    }

    predictedSlotsNames.zip(example.tokens).forEach { (predictedSlotName, token) ->

      if (predictedSlotName != null) {
        this.notNullSlots++
        if (predictedSlotName == token.slot.name) this.correctSlots++
      }
    }
  }

  /**
   * @return the evaluation statistics
   */
  private fun buildStats(): Statistics = Statistics(
    intents = buildStatsMetric(correct = correctIntents, outputTotal = totalIntents, goldTotal = totalIntents),
    slots = buildStatsMetric(correct = correctSlots, outputTotal = notNullSlots, goldTotal = totalSlots)
  )

  /**
   * @param correct the number of correct elements
   * @param outputTotal the number of output elements
   * @param goldTotal the number of gold elements
   *
   * @return statistics (precision, recall and F1 score) about the comparison of the output elements respect to
   *         the gold elements
   */
  private fun buildStatsMetric(correct: Int, outputTotal: Int, goldTotal: Int): Statistics.StatsMetric {

    val correctDouble: Double = correct.toDouble()

    return Statistics.StatsMetric(
      precision = if (outputTotal > 0) correctDouble / outputTotal else 0.0,
      recall = if (goldTotal > 0) correctDouble / goldTotal else 0.0,
      f1Score = if (outputTotal > 0 && goldTotal > 0) 2 * correctDouble / (outputTotal + goldTotal) else 0.0
    )
  }
}
