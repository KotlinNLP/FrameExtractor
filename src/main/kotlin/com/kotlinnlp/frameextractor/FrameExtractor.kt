/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.classifier.FrameClassifier
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The frame extractor.
 *
 * @param classifier a frame classifier
 * @param tokenizer a neural tokenizer
 * @param sentenceEncoder a sentence encoder
 */
class FrameExtractor(
  private val classifier: FrameClassifier,
  private val tokenizer: NeuralTokenizer,
  private val sentenceEncoder: SentenceEncoder
) {

  /**
   * Extract intent frames from a given text.
   *
   * @param text the input text
   *
   * @return the list of intent frames extracted
   */
  fun extractFrames(text: String): List<Intent> =

    this.tokenizer.tokenize(text).map { sentence ->

      val tokensForms: List<String> = sentence.tokens.map { it.form }
      val tokenEncodings: List<DenseNDArray> = this.sentenceEncoder.encode(tokensForms)
      val classifierOutput: FrameClassifier.Output = this.classifier.forward(tokenEncodings)

      classifierOutput.buildIntent(tokensForms = tokensForms, intentsConfig = this.classifier.model.intentsConfiguration)
    }
}
