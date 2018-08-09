/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.objects

import com.beust.klaxon.JsonObject
import com.beust.klaxon.json

/**
 * A slot of an [Intent].
 *
 * @property name the slot name
 * @property tokens the list of tokens that compose this slot
 */
class Slot(val name: String, val tokens: List<Token>) {

  /**
   * A token that compose a slot.
   *
   * @property index the index of the token within the list of tokens of its sentence
   * @property score the classification score of the token as part of the slot
   */
  data class Token(val index: Int, val score: Double)

  /**
   * @param tokenForms the list of token forms of the input sentence
   *
   * @return the JSON representation of this slot
   */
  fun toJSON(tokenForms: List<String>): JsonObject = json {
    obj(
      "name" to this@Slot.name,
      "value" to this@Slot.tokens.joinToString(" ") { tokenForms[it.index] },
      "tokens" to array(this@Slot.tokens.map { obj("index" to it.index, "score" to it.score) })
    )
  }
}
