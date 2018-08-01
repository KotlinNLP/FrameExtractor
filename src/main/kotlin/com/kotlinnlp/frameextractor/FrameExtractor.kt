/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.kotlinnlp.frameextractor.classifier.FrameClassifier
import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The frame extractor.
 *
 * @param classifier a frame classifier
 */
class FrameExtractor(private val classifier: FrameClassifier) {

  /**
   * Extract an intent frame from a given sentence that has been analyzed by a
   * [com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser].
   *
   * @param lss a latent syntactic structure of a sentence
   *
   * @return the intent frame extracted
   */
  fun extractFrame(lss: LatentSyntacticStructure): Intent {

    val tokenEncodings: List<DenseNDArray> = lss.contextVectors.zip(lss.latentHeads).map { it.first.concatH(it.second) }
    val classifierOutput: FrameClassifier.Output = this.classifier.forward(tokenEncodings)

    return classifierOutput.buildIntent(tokenForms = lss.sentence.tokens.map { it.form })
  }
}
