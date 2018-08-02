/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

import com.beust.klaxon.JsonObject
import com.beust.klaxon.json
import java.io.Serializable

/**
 * An intent.
 *
 * @property name the intent name
 * @property slots
 */
data class Intent(val name: String, val slots: List<Slot>) {

  /**
   * An intent configuration.
   *
   * @property name the intent name
   * @property slots the list of configurations of all the possible slots that can be associated to this intent
   */
  data class Configuration(val name: String, val slots: List<Slot.Configuration>) : Serializable {

    companion object {

      /**
       * Private val used to serialize the class (needed by Serializable).
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L
    }

    /**
     * The list of all the possible slots names of this intent.
     */
    val slotNames: List<String> by lazy { this.slots.map { it.name } }
  }

  /**
   * @return the JSON representation of this intent
   */
  fun toJSON(): JsonObject = json {
    obj(
      "name" to this@Intent.name,
      "slots" to array(this@Intent.slots.map { obj(it.name to it.value) }),
      "distribution" to this@Intent.distribution.toJSON()
    )
  }
}
