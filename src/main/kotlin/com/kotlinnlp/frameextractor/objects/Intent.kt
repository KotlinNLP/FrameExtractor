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
 * An intent.
 *
 * @property name the intent name
 * @property slots the list of slots of the intent
 * @property score the intent score (of prediction)
 */
data class Intent(val name: String, val slots: List<Slot>, val score: Double) {

  /**
   * The configuration of an [Intent].
   * It describes its name and its possible slots.
   *
   * @property name the intent name
   * @property slots the list of all the possible slot names that can be associated to this intent
   */
  data class Configuration(val name: String, val slots: List<String>) : Serializable {

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
    }

    /**
     * @param slotName the name of a possible slot of this intent
     *
     * @return the index of the slot with the given name within the possible slots defined in this configuration
     */
    fun getSlotIndex(slotName: String): Int = this.slots.indexOfFirst { it == slotName }
  }

  /**
   * @return the JSON representation of this intent
   */
  fun toJSON(): JsonObject = json {
    obj(
      "name" to this@Intent.name,
      "slots" to array(this@Intent.slots.map { it.toJSON() }),
      "score" to this@Intent.score
    )
  }
}
