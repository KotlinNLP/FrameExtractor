/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.objects.Distribution
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.frameextractor.objects.Slot
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The frame extractor.
 * It is a neural processor that given an encoded sentence (as list of encoded tokens) calculates a probability
 * distribution of the intents expressed by the sentence, together with the slot classifications associated to the
 * tokens.
 *
 * @property model the frame extractor model
 * @property propagateToInput whether to propagate errors to the input during the backward (default = false)
 * @property id an identifier of this frame extractor (useful when included in a pool, default = 0)
 */
class FrameExtractor(
  val model: FrameExtractorModel,
  override val propagateToInput: Boolean = false,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  FrameExtractor.Output, // OutputType
  FrameExtractor.Output, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  FrameExtractorParameters // ParamsErrorsType
  > {

  /**
   * A temporary slot with a mutable list of tokens.
   *
   * @property index the slot index
   * @property tokens the list of tokens that compose this slot
   */
  private data class TmpSlot(val index: Int, val tokens: MutableList<Slot.Token>)

  /**
   * The [FrameExtractor] output.
   *
   * @property intentsDistribution the distribution array of the intents
   * @property slotsClassifications the list of classifications of the slots, one per token
   */
  inner class Output(
    val intentsDistribution: DenseNDArray,
    val slotsClassifications: List<DenseNDArray>
  ) {

    /**
     * The list of intents configurations of the frame extractor related to this output.
     */
    private val intentsConfig: List<Intent.Configuration> = this@FrameExtractor.model.intentsConfiguration

    /**
     * Build a [Distribution] of the intents from this frame extractor output.
     *
     * @return the intents distribution
     */
    fun buildDistribution(): Distribution =
      Distribution(map = (0 until this.intentsDistribution.length).associate { i ->
        this.intentsConfig[i].name to this.intentsDistribution[i]
      })

    /**
     * Build an [Intent] from this frame extractor output.
     *
     * @return the intent interpreted from this output
     */
    fun buildIntent(): Intent {

      val intentIndex: Int = this.intentsDistribution.argMaxIndex()
      val intentConfig: Intent.Configuration = this.intentsConfig[intentIndex]
      val slotsOffset: Int = if (intentIndex > 0) this.intentsConfig.subList(0, intentIndex).sumBy { it.slots.size } else 0

      return Intent(
        name = intentConfig.name,
        slots = this.buildSlots(intentConfig = intentConfig, slotsOffset = slotsOffset),
        score = this.intentsDistribution[intentIndex]
      )
    }

    /**
     * @param intentConfig the intent configuration from which to extract the slots information
     * @param slotsOffset the offset of slots indices from which this intent starts in the whole list
     *
     * @return the list of slots interpreted from this output
     */
    private fun buildSlots(intentConfig: Intent.Configuration, slotsOffset: Int): List<Slot> {

      val slotsFound = mutableListOf<TmpSlot>()
      val slotsRange: IntRange = slotsOffset until (slotsOffset + intentConfig.slots.size)

      this.slotsClassifications.forEachIndexed { tokenIndex, classification ->

        val argMaxIndex: Int = classification.argMaxIndex()
        val slotIndex: Int = argMaxIndex / 2

        val token = Slot.Token(index = tokenIndex, score = classification[argMaxIndex])

        if (argMaxIndex % 2 == 0)
        // Beginning
          slotsFound.add(TmpSlot(index = slotIndex, tokens = mutableListOf(token)))
        else
        // Inside
          slotsFound.lastOrNull()?.let {
            if (it.index == slotIndex && it.tokens.last().index == tokenIndex - 1) it.tokens.add(token)
          }
      }

      return slotsFound
        .asSequence()
        .filter { it.index in slotsRange }
        .map { Slot(name = intentConfig.slots[it.index - slotsOffset], tokens = it.tokens) }
        .filter { it.name != Intent.Configuration.NO_SLOT_NAME }
        .toList()
    }
  }

  /**
   * The dropout is not useful for this processor because it has encodings as input and they make sense if used in
   * their original form.
   */
  override val useDropout: Boolean = false

  /**
   * The BiRNN1 encoder.
   */
  private val biRNNEncoder1 = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN1,
    propagateToInput = this.propagateToInput,
    useDropout = false)

  /**
   * The BiRNN2 encoder.
   */
  private val biRNNEncoder2 = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN2,
    propagateToInput = this.propagateToInput,
    useDropout = false)

  /**
   * The FF neural processor that decodes the intent.
   */
  private val intentProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.intentNetwork,
    propagateToInput = true,
    useDropout = false)

  /**
   * The FF batch processor that decodes the slots of an intent.
   */
  private val slotsProcessor = BatchFeedforwardProcessor<DenseNDArray>(
    neuralNetwork = this.model.slotsNetwork,
    propagateToInput = true,
    useDropout = false)

  /**
   * Calculate the distribution scores of the intents and the slots.
   *
   * @param input a list of the token encodings of a sentence
   *
   * @return a data class containing the distribution scores of the intents and the slots
   */
  override fun forward(input: List<DenseNDArray>): Output {

    val h1List: List<DenseNDArray> = this.biRNNEncoder1.forward(input)
    val h2List: List<DenseNDArray> = this.biRNNEncoder2.forward(input)

    val h1IntentInput: DenseNDArray = this.biRNNEncoder1.getLastOutput(copy = false).let { it.first.concatV(it.second) }
    val h2IntentInput: DenseNDArray = this.biRNNEncoder2.getLastOutput(copy = false).let { it.first.concatV(it.second) }

    // Attention: [h2, h1] inverted order!
    val slotsInputs: List<DenseNDArray> = h1List.zip(h2List).map { it.second.concatV(it.first) }

    return Output(
      intentsDistribution = this.intentProcessor.forward(h1IntentInput.concatV(h2IntentInput)).copy(),
      slotsClassifications = this.classifySlots(slotsInputs)
    )
  }

  /**
   * Execute the backward of the neural components given the output errors.
   *
   * @param outputErrors a data class containing the errors of the intent and slots distribution scores
   */
  override fun backward(outputErrors: Output) {

    val (h1IntentErrors, h2IntentErrors) = this.intentProcessor.let {
      it.backward(outputErrors.intentsDistribution)
      it.getInputErrors(copy = false).halfSplit()
    }

    val (h1IntentErrorsL2R, h1IntentErrorsR2L) = h1IntentErrors.halfSplit()
    val (h2IntentErrorsL2R, h2IntentErrorsR2L) = h2IntentErrors.halfSplit()

    val (h2Errors, h1Errors) = this.backwardSlotsErrors(outputErrors.slotsClassifications) // [h2, h1] inverted order!

    this.partialSum(h1Errors.last(), h1IntentErrorsL2R)
    this.partialSum(h1Errors.first(), h1IntentErrorsR2L, fromEnd = true)

    this.partialSum(h2Errors.last(), h2IntentErrorsL2R)
    this.partialSum(h2Errors.first(), h2IntentErrorsR2L, fromEnd = true)

    this.biRNNEncoder1.backward(h1Errors)
    this.biRNNEncoder2.backward(h2Errors)
  }

  /**
   * Get the list of input errors (they are always a copy).
   *
   * @param copy parameter inherited from the [NeuralProcessor] but without effect actually
   *
   * @return the list of input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    this.biRNNEncoder1.getInputErrors(copy = false)
      .zip(this.biRNNEncoder2.getInputErrors(copy = false))
      .map { it.first.sum(it.second) }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of this frame extractor parameters
   */
  override fun getParamsErrors(copy: Boolean) = FrameExtractorParameters(
    biRNN1Params = this.biRNNEncoder1.getParamsErrors(copy),
    biRNN2Params = this.biRNNEncoder2.getParamsErrors(copy),
    intentNetworkParams = this.intentProcessor.getParamsErrors(copy),
    slotsNetworkParams = this.slotsProcessor.getParamsErrors(copy)
  )

  /**
   * Get the offset index from which the slots of a given intent start, within the concatenation of all the possible
   * intents slots.
   *
   * @param intentName the name of an intent
   *
   * @return the offset of the given intent slots
   */
  fun getSlotsOffset(intentName: String): Int =
    this.model.intentsConfiguration
      .subList(0, this.model.intentsConfiguration.indexOfFirst { it.name == intentName })
      .sumBy { it.slots.size }

  /**
   * Sum a smaller dense array to a bigger dense array element-wise in-place, aligning them to the first or the last
   * index.
   *
   * @param bigArray the bigger dense array
   * @param smallArray the smaller dense array
   * @param fromEnd align the arrays to the last index if true, otherwise to the first index (the default)
   */
  private fun partialSum(bigArray: DenseNDArray, smallArray: DenseNDArray, fromEnd: Boolean = false) {

    (0 until smallArray.length).forEach { i ->

      val bigIndex: Int = if (fromEnd) bigArray.length - i - 1 else i

      bigArray[bigIndex] = bigArray[bigIndex] + smallArray[i]
    }
  }

  /**
   * Execute the backward of the slots processor.
   * Return a pair containing two lists of input errors, which are parallel and are split in the two components (h2 and
   * h1 respectively) that compose the inputs.
   *
   * @param slotsErrors the list of slots distributions errors
   *
   * @return a pair containing two parallel lists of slots processor input errors (h2, h1)
   */
  private fun backwardSlotsErrors(slotsErrors: List<DenseNDArray>): Pair<List<DenseNDArray>, List<DenseNDArray>> {

    val slotsInputSize: Int = this.model.biRNN1.outputSize + this.model.biRNN2.outputSize

    this.slotsProcessor.backward(slotsErrors)

    return this.slotsProcessor.getInputErrors(copy = false)
      .map { it.splitV(this.model.slotsNetwork.outputSize, slotsInputSize)[1].halfSplit() }
      .unzip()
  }

  /**
   * Split this dense array in two components, each with halved length.
   *
   * @return the two half components of this dense array
   */
  private fun DenseNDArray.halfSplit(): Pair<DenseNDArray, DenseNDArray> =
    this.splitV(this.length / 2).let { it[0] to it[1] }

  /**
   * @param slotsInputs the input array used to classify the intent slots, one per token
   *
   * @return the list of intent slots classification
   */
  private fun classifySlots(slotsInputs: List<DenseNDArray>): List<DenseNDArray> {

    var prevClass: Int? = null

    return slotsInputs.mapIndexed { i, slotsInput ->

      val prevClassBinary: DenseNDArray = prevClass?.let {
        DenseNDArrayFactory.oneHotEncoder(length = this.model.slotsNetwork.outputSize, oneAt = it)
      } ?: DenseNDArrayFactory.zeros(shape = Shape(this.model.slotsNetwork.outputSize))

      val input: List<DenseNDArray> = listOf(prevClassBinary.concatV(slotsInput))
      val classification: DenseNDArray = this.slotsProcessor.forward(input, continueBatch = i > 0).first()

      prevClass = classification.argMaxIndex()

      classification.copy()
    }
  }
}
