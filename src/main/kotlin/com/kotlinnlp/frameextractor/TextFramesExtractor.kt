/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.objects.Distribution
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuraltokenizer.Sentence as TokenizerSentence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The extractor of frames from a sentence of form tokens.
 *
 * @param model the text frames extractor model
 */
class TextFramesExtractor(private val model: TextFrameExtractorModel) {

  /**
   * A frame extracted.
   *
   * @property intent the intent frame extracted
   * @property distribution the distribution of [intent] classification
   */
  data class Frame(val intent: Intent, val distribution: Distribution)

  /**
   * A frame extractor built with the given [model].
   */
  internal val extractor = FrameExtractor(this.model.frameExtractor)

  /**
   * The encoder of the input tokens.
   */
  private val encoder: TokensEncoder<FormToken, Sentence<FormToken>> =
    this.model.tokensEncoder.buildEncoder(useDropout = false)

  /**
   * Extract intent frames from a sentence of form tokens.
   *
   * @param sentence a sentence of form tokens
   *
   * @return the frame extracted
   */
  fun extractFrames(sentence: Sentence<FormToken>): FrameExtractor.Output {

    val tokensEncodings: List<DenseNDArray> = this.encoder.forward(sentence)

    return this.extractor.forward(tokensEncodings)
  }
}
