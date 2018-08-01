/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.classifier

import com.kotlinnlp.simplednn.core.arrays.UpdatableArray
import com.kotlinnlp.simplednn.core.optimizer.IterableParams

/**
 * The [FrameClassifier] parameters.
 */
class FrameClassifierParameters : IterableParams<FrameClassifierParameters>() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  override val paramsList: List<UpdatableArray<*>>
    get() = TODO("not implemented")

  override fun copy(): FrameClassifierParameters {
    TODO("not implemented")
  }
}
