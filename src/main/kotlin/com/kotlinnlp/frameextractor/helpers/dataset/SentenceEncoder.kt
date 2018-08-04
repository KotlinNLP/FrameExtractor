/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers.dataset

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * An encoder of the tokens of a sentence.
 */
interface SentenceEncoder {

  /**
   * Encode the tokens forms.
   *
   * @param tokensForms the list of the tokens forms of a sentence
   *
   * @return the list of encodings, one per token
   */
  fun encode(tokensForms: List<String>): List<DenseNDArray>
}
