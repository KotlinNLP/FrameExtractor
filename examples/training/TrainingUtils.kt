/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatFeedforwardMerge
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.MirrorConverter
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel

/**
 * Add the all the tokens form to the embeddings map.
 *
 * @param sentences the list of sentences
 */
internal fun EmbeddingsMap<String>.addAll(sentences: List<Sentence<FormToken>>) {

  sentences.forEach { sentence ->
    sentence.tokens.forEach { token ->
      if (token.normalizedForm !in this)
        this.set(token.normalizedForm) // random initialized
    }
  }
}

/**
 * Build an [EnsembleTokensEncoderModel] composed by an embeddings encoder and an LSS encoder.
 *
 * @param preTrainedEmbeddingsMap an embeddings map by dictionary with pre-trained embeddings
 * @param emptyEmbeddingsMap an empty embeddings map to be trained
 * @param optimizeEmbeddings whether to optimize and serialize the embeddings
 *
 * @return a new tokens encoder model
 */
internal fun buildTokensEncoderModel(
  preTrainedEmbeddingsMap: EmbeddingsMap<String>,
  emptyEmbeddingsMap: EmbeddingsMap<String>,
  optimizeEmbeddings: Boolean
): TokensEncoderModel<FormToken, Sentence<FormToken>> {

  val preTrainedEmbeddingsEncoder: EmbeddingsEncoderModel<FormToken, Sentence<FormToken>> = if (optimizeEmbeddings)
    EmbeddingsEncoderModel
      .Base(embeddingsMap = preTrainedEmbeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())
  else
    EmbeddingsEncoderModel
      .Transient(embeddingsMap = preTrainedEmbeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())

  val trainableEmbeddingsEncoder: EmbeddingsEncoderModel<FormToken, Sentence<FormToken>> =
    EmbeddingsEncoderModel.Base(embeddingsMap = emptyEmbeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())

  return EnsembleTokensEncoderModel(
    components = listOf(
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(model = preTrainedEmbeddingsEncoder, converter = MirrorConverter()),
        trainable = optimizeEmbeddings),
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(model = trainableEmbeddingsEncoder, converter = MirrorConverter()),
        trainable = false)
    ),
    outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = 100)
  )
}
