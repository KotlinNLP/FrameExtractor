/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The [TextFramesExtractor] model.
 *
 * @property frameExtractor the model of the frame extractor
 * @property tokensEncoder the model of a tokens encoder to encode the input
 */
class TextFrameExtractorModel(
  val frameExtractor: FrameExtractorModel,
  val tokensEncoder: TokensEncoderModel<FormToken, Sentence<FormToken>>
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [TextFrameExtractorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [TextFrameExtractorModel]
     *
     * @return the [TextFrameExtractorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): TextFrameExtractorModel = Serializer.deserialize(inputStream)
  }

  /**
   * The name of this model.
   */
  val name: String = this.frameExtractor.name

  /**
   * Serialize this [TextFrameExtractorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [TextFrameExtractorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
