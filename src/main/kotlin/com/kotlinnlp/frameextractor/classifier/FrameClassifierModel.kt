/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.classifier

import com.kotlinnlp.frameextractor.Intent
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [FrameClassifier] parameters.
 *
 * @property intentsConfiguration the list of all the possible intents managed by this classifier
 */
class FrameClassifierModel(val intentsConfiguration: List<Intent.Configuration>) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [FrameClassifierModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [FrameClassifierModel]
     *
     * @return the [FrameClassifierModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): FrameClassifierModel = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [FrameClassifierModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [FrameClassifierModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
