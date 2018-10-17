/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatMerge
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [FrameExtractor] parameters.
 *
 * @property name the name of this model (it should be unique, used to distinguish it among more models)
 * @property intentsConfiguration the list of all the possible intents managed by this frame extractor
 * @param tokenEncodingSize
 * @param biRNNHiddenSize
 * @param biRNNHiddenActivation
 * @param biRNNConnectionType
 * @param slotsHiddenSize
 * @param slotsConnectionType
 */
class FrameExtractorModel(
  val name: String,
  val intentsConfiguration: List<Intent.Configuration>,
  tokenEncodingSize: Int,
  biRNNHiddenSize: Int,
  biRNNHiddenActivation: ActivationFunction? = Tanh(),
  biRNNConnectionType: LayerType.Connection = LayerType.Connection.LSTM,
  slotsHiddenSize: Int,
  slotsConnectionType: LayerType.Connection = LayerType.Connection.LSTM
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [FrameExtractorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [FrameExtractorModel]
     *
     * @return the [FrameExtractorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): FrameExtractorModel = Serializer.deserialize(inputStream)
  }

  /**
   *
   */
  val biRNN1 = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = tokenEncodingSize,
    hiddenSize = biRNNHiddenSize,
    hiddenActivation = biRNNHiddenActivation,
    recurrentConnectionType = biRNNConnectionType,
    dropout = 0.0, // the input is an encoding, it makes sense as complete numerical vector
    outputMergeConfiguration = ConcatMerge())

  /**
   *
   */
  val biRNN2 = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = tokenEncodingSize,
    hiddenSize = biRNNHiddenSize,
    hiddenActivation = biRNNHiddenActivation,
    recurrentConnectionType = biRNNConnectionType,
    dropout = 0.0, // the input is an encoding, it makes sense as complete numerical vector
    outputMergeConfiguration = ConcatMerge())

  /**
   *
   */
  val intentNetwork = NeuralNetwork(
    layersConfiguration = listOf(
      LayerInterface(
        size = 2 * this.biRNN1.hiddenSize + 2 * this.biRNN2.hiddenSize, // always the concatenation of the last outputs
        type = LayerType.Input.Dense),
      LayerInterface(
        size = this.intentsConfiguration.size,
        connectionType = LayerType.Connection.Feedforward,
        activationFunction = Softmax())
    )
  )

  /**
   *
   */
  val slotsRNN: NeuralNetwork

  /**
   *
   */
  val params: FrameExtractorParameters

  init {

    // There is a 2 x factor because it includes Beginning + Inside for each slot class.
    val slotsNetworkOutputSize: Int = 2 * this.intentsConfiguration.sumBy { it.slots.size }

    this.slotsRNN = NeuralNetwork(
      layersConfiguration = listOf(
        LayerInterface(
          size = this.biRNN1.outputSize + this.biRNN2.outputSize,
          type = LayerType.Input.Dense),
        LayerInterface(
          size = slotsHiddenSize,
          connectionType = slotsConnectionType,
          activationFunction = Tanh()),
        LayerInterface(
          size = slotsNetworkOutputSize,
          connectionType = LayerType.Connection.Feedforward,
          activationFunction = Softmax())
      ))

    this.params = FrameExtractorParameters(
      biRNN1Params = this.biRNN1.model,
      biRNN2Params = this.biRNN2.model,
      intentNetworkParams = this.intentNetwork.model,
      slotsRNNParams = this.slotsRNN.model
    )
  }

  /**
   * Serialize this [FrameExtractorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [FrameExtractorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
