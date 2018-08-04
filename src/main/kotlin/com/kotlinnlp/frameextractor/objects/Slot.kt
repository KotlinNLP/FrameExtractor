/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.objects

import com.beust.klaxon.JsonObject
import com.beust.klaxon.json
import java.io.Serializable

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
   * A slot configuration.
   *
   * @property name the slot name
   * @property required whether this slot is required or not
   * @property default the default value of this slot in case it is not required
   */
  data class Configuration(val name: String, val required: Boolean, val default: Any? = null) : Serializable {

    companion object {

      /**
       * Private val used to serialize the class (needed by Serializable).
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L

      /**
       * The name used to generate the slot for tokens that actually do not represent a slot of the intent.
       */
      const val NO_SLOT_NAME = "NoSlot"

      /**
       * The slot of tokens that actually do not represent a slot of the intent.
       */
      val noSlot: Configuration get() = Configuration(name = NO_SLOT_NAME, required = false)
    }
  }

  /**
   * @return the JSON representation of this slot
   */
  fun toJSON(): JsonObject = json {
    obj(
      "name" to this@Slot.name,
      "tokens" to array(this@Slot.tokens.map { obj("index" to it.index, "score" to it.score) })
    )
  }
}
