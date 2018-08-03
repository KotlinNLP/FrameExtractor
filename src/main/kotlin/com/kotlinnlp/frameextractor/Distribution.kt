/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.json

/**
 * The distribution of the prediction scores used to select a class during its generation.
 *
 * @property map a map of class names associated to the related prediction scores
 */
data class Distribution(val map: Map<String, Double>) {

  /**
   * @return a JSON representation of this distribution
   */
  fun toJSON(): JsonArray<JsonObject> = json {
    @Suppress("UNCHECKED_CAST")
    array(
      this@Distribution.map.entries
        .sortedByDescending { it.value }
        .map { obj("score" to it.value, "name" to it.key) }
    ) as JsonArray<JsonObject>
  }
}
