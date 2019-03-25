/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers.dataset

import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.Sentence as LDSentence
import com.kotlinnlp.frameextractor.objects.Slot as IntentSlot

/**
 * A dataset that can be read from a JSON file and it is used to create an [EncodedDataset].
 *
 * @property configuration the list of configurations of all the possible intents in this dataset
 * @property examples the list of examples
 */
data class Dataset(
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
   * @property intent the name of the intent that the example represents
   * @property sentence the sentence of the example
   */
  data class Example(val intent: String, val sentence: Sentence) {

    /**
     * The sentence of the example
     */
    data class Sentence(override val tokens: List<Token>) : LDSentence<FormToken>

    /**
     * A token of the sentence.
     *
     * @property form the token form
     * @property slot the intent slot that this token is part of
     */
    data class Token(override val form: String, val slot: Slot) : FormToken

    /**
     * The token slot.
     *
     * @property name the slot name
     * @property iob the IOB tag of the token within its slot ("B" beginning or "I" inside)
     */
    data class Slot(val name: String, val iob: IOBTag) {

      companion object {

        /**
         * The slot of tokens that actually do not represent a slot of the intent.
         * It is automatically created when a [Token] has null 'slot' property.
         */
        val noSlot: Slot get() = Slot(name = Intent.Configuration.NO_SLOT_NAME, iob = IOBTag.Beginning)
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
    fun fromFile(filePath: String): Dataset {

      val jsonDataset: JsonObject = Parser().parse(filePath) as JsonObject

      return Dataset(
        configuration = jsonDataset.array<JsonObject>("intents")!!.map { intent ->
          Intent.Configuration(
            name = intent.string("name")!!,
            slots = intent.array("slots")!!)
        },
        examples = jsonDataset.array<JsonObject>("examples")!!.map { example ->
          Example(
            intent = example.string("intent")!!,
            sentence = Example.Sentence(
              tokens = example.array<JsonObject>("tokens")!!.map { token ->
                Example.Token(
                  form = token.string("form")!!,
                  slot = token.obj("slot")?.let { slot ->
                    Example.Slot(
                      name = slot.string("name")!!,
                      iob = slot.string("iob")!!.let { iob ->
                        when (iob) {
                          "B" -> IOBTag.Beginning
                          "I" -> IOBTag.Inside
                          else -> throw RuntimeException("Invalid IOB tag annotation: $iob")
                        }
                      })
                  } ?: Example.Slot.noSlot
                )
              }
            )
          )
        }
      )
    }
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

      example.sentence.tokens
        .firstOrNull {
          it.slot.name != Intent.Configuration.NO_SLOT_NAME &&
            it.slot.name !in slotNamesByIntent.getValue(example.intent)
        }?.let {
          throw InvalidExample(
            index = exampleIndex,
            message = "Invalid slot name for intent '${example.intent}': '${it.slot.name}'")
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

      val slotsSet: Set<String> = it.slots.toSet()

      if (slotsSet.size != it.slots.size)
        throw InvalidIntentConfiguration("Intent '${it.name}': slot names must be unique.")

      it.name to slotsSet
    }

    if (slotNamesByIntent.keys.size != this.configuration.size)
      throw InvalidConfiguration("Intent names must be unique.")

    return slotNamesByIntent
  }
}
