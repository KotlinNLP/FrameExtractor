/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils

import com.kotlinnlp.frameextractor.classifier.helpers.dataset.SentenceEncoder
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.BaseToken
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.LSSEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder

/**
 * An encoder of the tokens of a sentence using the Latent Syntactic Structure of an [LSSEncoder] and words embeddings.
 *
 * @param wordEmbeddingsEncoder a word embeddings encoder
 * @param lssEncoder a Latent Syntactic Structure encoder
 * @param preprocessor a sentence preprocessor
 */
class LSSEmbeddingsEncoder(
  private val wordEmbeddingsEncoder: EmbeddingsEncoder,
  private val lssEncoder: LSSEncoder,
  private val preprocessor: SentencePreprocessor
) : SentenceEncoder {

  /**
   * The size of the tokens encodings (word embedding + context vectors + latent head representation).
   * Note: the context vectors size is equal to the latent head representations size.
   */
  val encodingSize: Int =
    this.wordEmbeddingsEncoder.model.tokenEncodingSize + 2 * lssEncoder.contextEncoder.model.contextEncodingSize

  /**
   * Encode the token forms concatenating word embeddings, latent head representations and context vectors.
   *
   * @param tokensForms the list of the tokens forms of a sentence
   *
   * @return the list of encodings, one per token
   */
  override fun encode(tokensForms: List<String>): List<DenseNDArray> {

    val sentence: ParsingSentence = this.buildSentence(tokensForms)
    val lss = lssEncoder.encode(sentence)

    @Suppress("UNCHECKED_CAST")
    val wordEncodings: List<DenseNDArray> = wordEmbeddingsEncoder.forward(sentence as Sentence<Token>)

    val lssEncodings: List<DenseNDArray> = lss.latentHeads.zip(lss.contextVectors) {
      latentHead, contextVector -> latentHead.concatV(contextVector)
    }

    return lssEncodings.zip(wordEncodings) { lssEncoding, wordEncoding -> lssEncoding.concatV(wordEncoding) }
  }

  /**
   * @param tokensForms the list of tokens forms
   *
   * @return a parsing sentence built with the given tokens forms
   */
  private fun buildSentence(tokensForms: List<String>): ParsingSentence {

    var tokenPosition = 0

    return this.preprocessor.process(BaseSentence(
      position = Position(index = 0, start = 0, end = tokensForms.sumBy { it.length + 1 } - 1),
      tokens = tokensForms.mapIndexed { i, form ->

        val baseToken = BaseToken(
          id = i,
          form = form,
          position = Position(index = i, start = tokenPosition, end = tokenPosition + form.length))

        tokenPosition += form.length + 1

        baseToken
      }
    ))
  }
}
