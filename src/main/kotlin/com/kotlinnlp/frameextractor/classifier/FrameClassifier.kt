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
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

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
      val slotsOffset: Int = if (intentIndex > 0)
        intentsConfig.subList(0, intentIndex - 1).sumBy { it.slots.size }
      else
        0

      return Intent(
        name = intentConfig.name,
        slots = this.buildSlots(tokenForms = tokenForms, intentConfig = intentConfig, slotsOffset = slotsOffset)
      )
    }

    /**
     * @param tokenForms the list of tokens forms
     * @param intentConfig the intent configuration from which to extract the slots information
     * @param slotsOffset the offset of slots indices from which this intent starts in the whole list
     *
     * @return the list of slots interpreted from this output
     */
    private fun buildSlots(tokenForms: List<String>, intentConfig: Intent.Configuration, slotsOffset: Int): List<Slot> {

      val slotsFound = mutableListOf<Triple<Int, Int, StringBuffer>>()

      this.slotsClassifications.forEachIndexed { tokenIndex, classification ->

        val tokenForm: String = tokenForms[tokenIndex]
        val argMaxIndex: Int = classification.argMaxIndex()
        val slotIndex: Int = argMaxIndex / 2
        val slotIOB: IOBTag = if (argMaxIndex % 2 == 0) IOBTag.Beginning else IOBTag.Inside

        if (slotIOB == IOBTag.Beginning)
          slotsFound.add(Triple(tokenIndex, slotIndex, StringBuffer(tokenForm)))
        else
          slotsFound.lastOrNull()?.let {
            if (it.first == tokenIndex - 1 && it.second == slotIndex) it.third.append(" $tokenForm")
          }
      }

      return slotsFound.map {
        Slot(name = intentConfig.slotNames[it.second - slotsOffset], value = it.third.toString())
      }
    }
  }

  /**
   *
   */
  private val biRNNEncoder1 = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN1,
    propagateToInput = false,
    useDropout = false)

  /**
   *
   */
  private val biRNNEncoder2 = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN2,
    propagateToInput = false,
    useDropout = false)

  /**
   *
   */
  private val intentProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.intentNetwork,
    propagateToInput = true,
    useDropout = false)

  /**
   *
   */
  private val slotsProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.intentNetwork,
    propagateToInput = true,
    useDropout = false)

  /**
   *
   */
  override fun forward(input: List<DenseNDArray>): Output {

    val h1List: List<DenseNDArray> = this.biRNNEncoder1.forward(input)
    val h2List: List<DenseNDArray> = this.biRNNEncoder2.forward(input)

    val h1IntentInput: DenseNDArray = this.biRNNEncoder1.getLastOutput(copy = false).let { it.first.concatV(it.second) }
    val h2IntentInput: DenseNDArray = this.biRNNEncoder2.getLastOutput(copy = false).let { it.first.concatV(it.second) }

    val h1SlotsInputs: List<DenseNDArray> = h1List.zip(h2List).map { it.first.concatH(it.second) }
    val h2SlotsInputs: List<DenseNDArray> = h1List.zip(h2List).map { it.second.concatH(it.first) }

    return Output(
      intentsDistribution = this.intentProcessor.forward(h1IntentInput.concatV(h2IntentInput)),
      slotsClassifications = this.classifySlots(h1SlotsInputs.zip(h2SlotsInputs) { h1s, h2s -> h1s.concatV(h2s) })
    )
  }

  /**
   *
   */
  override fun backward(outputErrors: Output) {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun getParamsErrors(copy: Boolean) = FrameClassifierParameters(
    biRNN1Params = this.biRNNEncoder1.getParamsErrors(),
    biRNN2Params = this.biRNNEncoder2.getParamsErrors(),
    intentNetworkParams = this.intentProcessor.getParamsErrors(),
    slotsNetworkParams = this.slotsProcessor.getParamsErrors()
  )

  /**
   *
   */
  private fun classifySlots(slotsInputs: List<DenseNDArray>): List<DenseNDArray> {

    var prevClass: Int? = null

    return slotsInputs.map { slotsInput ->

      val prevClassBinary: DenseNDArray = prevClass?.let {
        DenseNDArrayFactory.oneHotEncoder(length = this.model.slotsNetwork.outputSize, oneAt = it)
      } ?: DenseNDArrayFactory.zeros(shape = Shape(this.model.slotsNetwork.outputSize))

      val classification: DenseNDArray = this.slotsProcessor.forward(prevClassBinary.concatV(slotsInput))

      prevClass = classification.argMaxIndex()

      classification
    }
  }
}
