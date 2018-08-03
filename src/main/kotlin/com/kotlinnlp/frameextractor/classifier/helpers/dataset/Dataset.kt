/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.classifier.helpers.dataset

import com.beust.klaxon.Json
import com.beust.klaxon.Klaxon
import com.kotlinnlp.frameextractor.Intent
import com.kotlinnlp.frameextractor.Slot as IntentSlot
import java.io.File

/**
 * A dataset that can be read from a JSON file and it is used to create an [EncodedDataset].
 *
 * @property configuration the list of configurations of all the possible intents in this dataset
 * @property examples the list of examples
 */
data class Dataset(
  @Json(name = "intents")
  val configuration: List<Intent.Configuration>,
  val examples: List<Example>
) {

  /**
   * Raised when the configuration of the dataset is not valid.
   *
   * @param message the exception message
   */
  class InvalidConfiguration(message: String) : RuntimeException(message)

  /**
   * Raised when the configuration of an intent is not valid.
   *
   * @param message the exception message
   */
  class InvalidIntentConfiguration(message: String) : RuntimeException(message)

  /**
   * Raised when an example is not valid.
   *
   * @param index the example index
   * @param message the exception message
   */
  class InvalidExample(index: Int, message: String) : RuntimeException("[Example #$index] $message")

  /**
   * An example of the dataset.
   *
   * @property intent the name of the intent that this example represents
   * @property tokens the list of tokens that compose the sentence of this example
   */
  data class Example(val intent: String, val tokens: List<Token>) {

    /**
     * A token of the example.
     *
     * @property form the token form
     * @property slot the intent slot that this token is part of (null if it is not part of a slot)
     */
    data class Token(val form: String, val slot: Slot? = null)

    /**
     * The token slot.
     *
     * @property name the slot name
     * @property iob the IOB tag of the token within its slot ("B" beginning or "I" inside)
     */
    data class Slot(val name: String, val iob: String) {

      companion object {

        /**
         * The slot of tokens that actually do not represent a slot of the intent.
         * It is automatically created when a [Token] has null 'slot' property.
         */
        val noSlot: Slot get() = Slot(name = IntentSlot.Configuration.NO_SLOT_NAME, iob = "B")
      }

      /**
       * The IOB tag as enum class.
       */
      @Json(ignored = true)
      val iobTag: IOBTag = when (iob) {
        "B" -> IOBTag.Beginning
        "I" -> IOBTag.Inside
        else -> throw RuntimeException("Invalid IOB tag annotation: $iob")
      }
    }
  }

  companion object {

    /**
     * Load a [Dataset] from a JSON file with the following template:
     *
     * {
     *   "intents": [ // the configuration of the possible intents
     *      {
     *        "name": "INTENT_NAME",
     *        "slots": [ // the configuration of the possible slots of this intent
     *          {
     *            "name": "SLOT_NAME",
     *            "required": true,
     *            "default": 2 // it can be any base type
     *          }
     *        ]
     *      }
     *   ],
     *   "examples": [
     *     {
     *       "intent": "INTENT_NAME",
     *       "tokens": [
     *         {
     *           "form": "TOKEN_FORM",
     *           "slot": { // null if the token is not part of a slot
     *             "name": "SLOT_NAME",
     *             "iob": "B" // only "B" (beginning) or "I" (inside) values are accepted
     *           }
     *         }
     *       ]
     *     }
     *   ]
     * }
     *
     * @param filePath the file path of the JSON file
     *
     * @return the dataset read from the given file
     */
    fun fromFile(filePath: String): Dataset = Klaxon().parse<Dataset>(File(filePath))!!
  }

  /**
   * Check the validity of the dataset.
   */
  init {

    val slotNamesByIntent: Map<String, Set<String>> = this.checkConfiguration()

    this.examples.forEachIndexed { i, example ->

      val exampleIndex: Int = i + 1

      if (example.intent !in slotNamesByIntent)
        throw InvalidExample(index = exampleIndex, message = "Invalid intent name: '${example.intent}'")

      example.tokens.forEach {
        if (it.slot != null && it.slot.name !in slotNamesByIntent.getValue(example.intent))
          throw InvalidExample(index = exampleIndex, message = "Invalid slot name: '${it.slot.name}'")
      }
    }
  }

  /**
   * Check the validity of the dataset configuration.
   *
   * @throws InvalidConfiguration when the configuration is generically not valid
   * @throws InvalidIntentConfiguration when the configuration of an intent is not valid
   *
   * @return the possible intent names associated to the related set of possible slots names
   */
  private fun checkConfiguration(): Map<String, Set<String>> {

    val slotNamesByIntent: Map<String, Set<String>> = this.configuration.associate {

      val slotNames: Set<String> = it.slots.map { slot -> slot.name }.toSet()

      if (slotNames.size != it.slots.size)
        throw InvalidIntentConfiguration("Intent '${it.name}': slot names must be unique.")

      it.name to slotNames
    }

    if (slotNamesByIntent.keys.size != this.configuration.size)
      throw InvalidConfiguration("Intent names must be unique.")

    return slotNamesByIntent
  }
}
