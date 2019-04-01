/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.optimizer.IterableParams
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * The [FramesExtractorModel] parameters.
 *
 * @property biRNN1Params
 * @property biRNN2Params
 * @property intentNetworkParams
 * @property slotsNetworkParams
 */
class FrameExtractorParameters(
  val biRNN1Params: BiRNNParameters,
  val biRNN2Params: BiRNNParameters,
  val intentNetworkParams: StackedLayersParameters,
  val slotsNetworkParams: StackedLayersParameters
) : IterableParams<FrameExtractorParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of all the parameters of a [FramesExtractorModel].
   */
  override val paramsList: List<ParamsArray> =
    this.biRNN1Params.paramsList +
      this.biRNN2Params.paramsList +
      this.intentNetworkParams.paramsList +
      this.slotsNetworkParams.paramsList

  /**
   * @return new [FrameExtractorParameters] containing a copy of all the values of this
   */
  override fun copy() = FrameExtractorParameters(
    biRNN1Params = this.biRNN1Params.copy(),
    biRNN2Params = this.biRNN2Params.copy(),
    intentNetworkParams = this.intentNetworkParams.copy(),
    slotsNetworkParams = this.slotsNetworkParams.copy()
  )
}
