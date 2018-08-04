/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers.dataset

/**
 * The IOB tag.
 *
 * @property annotation the tag annotation
 */
enum class IOBTag(val annotation: String) {
  Inside("I"),
  Outside("O"),
  Beginning("B")
}
