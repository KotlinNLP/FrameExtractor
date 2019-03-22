/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package extract

import com.kotlinnlp.frameextractor.objects.Distribution
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.frameextractor.FrameExtractor
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuraltokenizer.Sentence as TokenizerSentence
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The extractor of frames from a sentence of form tokens.
 *
 * @param extractor a frame extractor
 * @param tokensEncoder a tokens encoder
 */
internal class TextFramesExtractor(
  private val extractor: FrameExtractor,
  private val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>>
) {

  /**
   * A frame extracted.
   *
   * @property intent the intent frame extracted
   * @property distribution the distribution of [intent] classification
   */
  data class Frame(val intent: Intent, val distribution: Distribution)

  /**
   * Extract intent frames from a sentence of form tokens.
   *
   * @param sentence a sentence of form tokens
   *
   * @return the frame extracted
   */
  fun extractFrames(sentence: Sentence<FormToken>): Frame {

    val tokenEncodings: List<DenseNDArray> = this.tokensEncoder.forward(sentence)
    val extractorOutput: FrameExtractor.Output = this.extractor.forward(tokenEncodings)

    return Frame(
      intent = extractorOutput.buildIntent(),
      distribution = extractorOutput.buildDistribution())
  }
}
