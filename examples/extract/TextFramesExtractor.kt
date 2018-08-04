/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package extract

import com.kotlinnlp.frameextractor.Distribution
import com.kotlinnlp.frameextractor.Intent
import com.kotlinnlp.frameextractor.LSSEmbeddingsEncoder
import com.kotlinnlp.frameextractor.classifier.FrameClassifier
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The extractor of frames from a text.
 *
 * @param classifier a frame classifier
 * @param tokenizer a neural tokenizer
 * @param sentenceEncoder a sentence encoder
 */
internal class TextFramesExtractor(
  private val classifier: FrameClassifier,
  private val tokenizer: NeuralTokenizer,
  private val sentenceEncoder: LSSEmbeddingsEncoder
) {

  /**
   * A frame extracted.
   *
   * @property intent the intent frame extracted
   * @property distribution the distribution of [intent] classification
   * @property sentence the intent sentence
   */
  data class Frame(val intent: Intent, val distribution: Distribution, val sentence: Sentence<FormToken>)

  /**
   * Extract intent frames from a text.
   *
   * @param text the input text
   *
   * @return the list of frames extracted
   */
  @Suppress("UNCHECKED_CAST")
  fun extractFrames(text: String): List<Frame> =

    this.tokenizer.tokenize(text).map {

      val sentence = it as Sentence<FormToken>
      val tokenEncodings: List<DenseNDArray> = this.sentenceEncoder.encode(tokensForms = sentence.tokens.map { it.form })
      val classifierOutput: FrameClassifier.Output = this.classifier.forward(tokenEncodings)

      Frame(
        intent = classifierOutput.buildIntent(),
        distribution = classifierOutput.buildDistribution(),
        sentence = sentence
      )
    }
}
