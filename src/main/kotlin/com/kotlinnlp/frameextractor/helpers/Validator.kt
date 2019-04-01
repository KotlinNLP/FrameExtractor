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
import com.kotlinnlp.frameextractor.TextFramesExtractorModel
import com.kotlinnlp.frameextractor.TextFramesExtractor
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.utils.stats.MetricCounter

/**
 * A helper to evaluate a [FrameExtractorModel].
 *
 * @param model the model to evaluate
 * @param dataset the validation dataset
 * @param verbose whether to print info about the validation progress (default = true)
 */
class Validator(
  private val model: TextFramesExtractorModel,
  private val dataset: Dataset,
  private val verbose: Boolean = true
) {

  /**
   * A frame extractor built with the given [model].
   */
  private val extractor = TextFramesExtractor(this.model)

  /**
   * The metric counters for each intent, associated by intent name.
   */
  private lateinit var intentMetrics: Map<String, MetricCounter>

  /**
   * The metric counter of the slots.
   */
  private lateinit var slotsMetric: MetricCounter

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

    return Statistics(intents = this.intentMetrics, slots = this.slotsMetric)
  }

  /**
   * Reset the metrics counters.
   */
  private fun reset() {

    this.intentMetrics = this.dataset.configuration.associate { it.name to MetricCounter() }
    this.slotsMetric = MetricCounter()
  }

  /**
   * Validate the [model] on a single example.
   *
   * @param example an example of the dataset
   */
  private fun validateExample(example: Dataset.Example) {

    val output: FrameExtractor.Output = this.extractor.extractFrames(example.sentence)

    val bestIntentIndex: Int = output.intentsDistribution.argMaxIndex()
    val intentConfig: Intent.Configuration = this.model.frameExtractor.intentsConfiguration[bestIntentIndex]
    val bestIntentName: String = intentConfig.name

    if (example.intent == bestIntentName) {

      this.intentMetrics.getValue(example.intent).truePos++

      this.validateSlots(
        example = example,
        possibleSlots = intentConfig.slots,
        slotsClassifications = output.slotsClassifications)

    } else {
      this.intentMetrics.getValue(example.intent).falsePos++
    }
  }

  /**
   * Validate the slots classified.
   *
   * @param example the example from which the slots have been classified
   * @param possibleSlots the list of possible slot names that can be associated to the example intent
   * @param slotsClassifications the list of slots classifications
   */
  private fun validateSlots(example: Dataset.Example,
                            possibleSlots: List<String>,
                            slotsClassifications: List<DenseNDArray>) {

    val intentSlotsOffset: Int = this.model.frameExtractor.getSlotsOffset(example.intent)
    val intentSlotsRange = intentSlotsOffset until (intentSlotsOffset + possibleSlots.size)

    val predictedSlotsNames: List<String?> = slotsClassifications.map {

      val bestSlotIndex: Int = it.argMaxIndex() / 2

      if (bestSlotIndex in intentSlotsRange) possibleSlots[bestSlotIndex - intentSlotsOffset] else null
    }

    predictedSlotsNames.zip(example.sentence.tokens).forEach {
      this.validateSlot(predicted = it.first, expected = it.second.slot.name)
    }
  }

  /**
   * Validate a slot classified.
   *
   * @param predicted the name of the predicted slot (null if it is not a slot of the predicted intent)
   * @param expected the name of the expected slot
   */
  private fun validateSlot(predicted: String?, expected: String) {

      if (predicted != null) {

        if (predicted == expected)
          this.slotsMetric.truePos++
        else
          this.slotsMetric.falsePos++

      } else {
        this.slotsMetric.falseNeg++
      }
  }
}
