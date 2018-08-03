/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.classifier.FrameClassifier
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The frame extractor.
 *
 * @param classifier a frame classifier
 * @param sentenceEncoder a sentence encoder
 */
class FrameExtractor(
  private val classifier: FrameClassifier,
  private val sentenceEncoder: SentenceEncoder
) {

  /**
   * Extract an intent frame from a given sentence.
   *
   * @param sentence the input sentence
   *
   * @return the intent frame extracted
   */
  fun extractFrame(sentence: Sentence<FormToken>): Intent {

    val tokensForms: List<String> = sentence.tokens.map { it.form }
    val tokenEncodings: List<DenseNDArray> = this.sentenceEncoder.encode(tokensForms)
    val classifierOutput: FrameClassifier.Output = this.classifier.forward(tokenEncodings)

    return classifierOutput.buildIntent(
      tokensForms = tokensForms,
      intentsConfig = this.classifier.model.intentsConfiguration)
  }
}
