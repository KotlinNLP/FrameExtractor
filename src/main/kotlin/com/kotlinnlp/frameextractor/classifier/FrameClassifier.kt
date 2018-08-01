/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.classifier

import com.kotlinnlp.frameextractor.IOBTag
import com.kotlinnlp.frameextractor.Intent
import com.kotlinnlp.frameextractor.Slot
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The frame classifier.
 * It is a neural processor that given an encoded sentence (as list of encoded tokens) calculates a probability
 * distribution of the intents expressed by the sentence, together with the slot classifications associated to the
 * tokens.
 */
class FrameClassifier(
  val model: FrameClassifierModel,
  override val useDropout: Boolean,
  override val propagateToInput: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  FrameClassifier.Output, // OutputType
  FrameClassifier.Output, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  FrameClassifierParameters // ParamsErrorsType
  > {

  /**
   * The [FrameClassifier] output.
   *
   * @property intentsDistribution the distribution array of the intents
   * @property slotsClassifications the list of classifications of the slots, one per token
   */
  data class Output(
    val intentsDistribution: DenseNDArray,
    val slotsClassifications: List<DenseNDArray>
  ) {

    /**
     * Build an [Intent] from this classifier output.
     *
     * @param tokenForms the list of tokens forms
     * @param intentsConfig the intents configuration from which to extract the intents information
     *
     * @return the intent interpreted from this output
     */
    fun buildIntent(tokenForms: List<String>, intentsConfig: List<Intent.Configuration>): Intent {

      val intentIndex: Int = this.intentsDistribution.argMaxIndex()
      val intentConfig: Intent.Configuration = intentsConfig[intentIndex]

      return Intent(
        name = intentConfig.name,
        slots = this.buildSlots(tokenForms = tokenForms, intentConfig = intentConfig)
      )
    }

    /**
     * @param tokenForms the list of tokens forms
     * @param intentConfig the intent configuration from which to extract the slots information
     *
     * @return the list of slots interpreted from this output
     */
    private fun buildSlots(tokenForms: List<String>, intentConfig: Intent.Configuration): List<Slot> {

      val slotsFound = mutableListOf<Triple<Int, StringBuffer, Int>>()

      this.slotsClassifications.forEachIndexed { tokenIndex, classification ->

        val tokenForm: String = tokenForms[tokenIndex]
        val argMaxIndex: Int = classification.argMaxIndex()
        val slotIndex: Int = argMaxIndex / 2
        val slotIOB: IOBTag = if (argMaxIndex % 2 == 0) IOBTag.Beginning else IOBTag.Inside

        if (slotIOB == IOBTag.Beginning)
          slotsFound.add(Triple(slotIndex, StringBuffer(tokenForm), tokenIndex))
        else
          slotsFound.lastOrNull()?.let {
            if (it.first == slotIndex && it.third == tokenIndex - 1) it.second.append(" $tokenForm")
          }
      }

      return slotsFound.map { Slot(name = intentConfig.slotNames[it.first], value = it.second.toString()) }
    }
  }

  override fun forward(input: List<DenseNDArray>): Output {
    TODO("not implemented")
  }

  override fun backward(outputErrors: Output) {
    TODO("not implemented")
  }

  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {
    TODO("not implemented")
  }

  override fun getParamsErrors(copy: Boolean): FrameClassifierParameters {
    TODO("not implemented")
  }
}
