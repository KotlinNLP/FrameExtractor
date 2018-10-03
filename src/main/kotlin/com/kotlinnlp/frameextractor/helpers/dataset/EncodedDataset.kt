/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers.dataset

import com.kotlinnlp.frameextractor.objects.Intent
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A dataset with the same structure of the [Dataset] read from file, but with encodings instead of forms in the
 * tokens of its examples.
 * It is used to train and validate a [com.kotlinnlp.frameextractor.FrameExtractor].
 *
 * @property configuration the list of configurations of all the possible intents in this dataset
 * @property examples the list of examples
 */
data class EncodedDataset(
  val configuration: List<Intent.Configuration>,
  val examples: List<Example>
) {

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
     * @property encoding the dense array that represents the encoding of the token
     * @property slot the intent slot that this token is part of (null if it is not part of a slot)
     */
    data class Token(val encoding: DenseNDArray, val slot: Dataset.Example.Slot)
  }

  /**
   * The input token.
   *
   * @property form the form of the token
   */
  private class InputToken(override val form: String) : FormToken

  /**
   * The input sentence.
   *
   * @property tokens the list of tokens
   */
  private class InputSentence(override val tokens: List<InputToken>) : Sentence<FormToken>

  /**
   * Factory object.
   */
  companion object {

    /**
     * Build an [EncodedDataset] from a [Dataset], encoding the tokens of its examples with a given [TokensEncoder].
     *
     * @param dataset a dataset
     * @param tokensEncoder a tokens encoder
     *
     * @return an encoded dataset
     */
    fun fromDataset(dataset: Dataset,
                    tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>>,
                    printProgress: Boolean = true): EncodedDataset {

      val progress = if (printProgress) ProgressIndicatorBar(dataset.examples.size) else null

      return EncodedDataset(
        configuration = dataset.configuration.map {
          it.copy(slots = it.slots + Intent.Configuration.NO_SLOT_NAME)
        },
        examples = dataset.examples.map {

          val tokensEncodings: List<DenseNDArray> = tokensEncoder.forward(buildSentence(it))

          progress?.tick()

          Example(
            intent = it.intent,
            tokens = it.tokens.zip(tokensEncodings).map { (token, encoding) ->
              Example.Token(encoding = encoding, slot = token.slot
                ?: Dataset.Example.Slot.noSlot)
            }
          )
        }
      )
    }

    /**
     * @param example a dataset example
     *
     * @return a new input sentence built from the given example
     */
    private fun buildSentence(example: Dataset.Example): Sentence<FormToken> = InputSentence(
      tokens = example.tokens.map { InputToken(it.form) })
  }
}
