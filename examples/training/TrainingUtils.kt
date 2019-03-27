/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.lssencoder.tokensencoder.LSSTokensEncoderModel
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.BaseToken
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatFeedforwardMerge
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.MirrorConverter
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter
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
 * @param preprocessor a sentence preprocessor
 * @param embeddingsMap an embeddings map by dictionary
 * @param lssModel the model of an LSS encoder
 * @param optimizeEmbeddings whether to optimize and serialize the embeddings
 *
 * @return a new tokens encoder model
 */
internal fun buildTokensEncoderModel(
  preprocessor: SentencePreprocessor,
  embeddingsMap: EmbeddingsMap<String>,
  lssModel: LSSModel<ParsingToken, ParsingSentence>,
  optimizeEmbeddings: Boolean
): TokensEncoderModel<FormToken, Sentence<FormToken>> {

  val embeddingsEncoder: EmbeddingsEncoderModel<FormToken, Sentence<FormToken>> = if (optimizeEmbeddings)
    EmbeddingsEncoderModel
      .Base(embeddingsMap = preTrainedEmbeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())
  else
    EmbeddingsEncoderModel
      .Transient(embeddingsMap = preTrainedEmbeddingsMap, embeddingKeyExtractor = NormWordKeyExtractor())

  val lssEncoder = LSSTokensEncoderModel(lssModel)

  return EnsembleTokensEncoderModel(
    components = listOf(
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(model = embeddingsEncoder, converter = MirrorConverter()),
        trainable = optimizeEmbeddings),
      EnsembleTokensEncoderModel.ComponentModel(
        model = TokensEncoderWrapperModel(model = lssEncoder, converter = FormSentenceConverter(preprocessor)),
        trainable = false)
    ),
    outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = 100)
  )
}

/**
 * The [SentenceConverter] from a sentence of form tokens.
 *
 * @param preprocessor a sentence preprocessor
 */
private class FormSentenceConverter(
  private val preprocessor: SentencePreprocessor
) : SentenceConverter<FormToken, Sentence<FormToken>, ParsingToken, ParsingSentence> {

  /**
   * @param sentence an input sentence
   *
   * @return a parsing sentence built with the given input sentence
   */
  override fun convert(sentence: Sentence<FormToken>): ParsingSentence =
    this.preprocessor.convert(this.buildBaseSentence(sentence))

  /**
   * @param sentence an input sentence
   *
   * @return a new base sentence built from the given input sentence
   */
  private fun buildBaseSentence(sentence: Sentence<FormToken>): BaseSentence {

    var tokenPosition = 0

    fun nextPosition(token: FormToken): Int {
      tokenPosition += token.form.length + 1
      return tokenPosition
    }

    return BaseSentence(
      id = 0,
      position = Position(index = 0, start = 0, end = sentence.tokens.sumBy { it.form.length + 1 }),
      tokens = sentence.tokens.mapIndexed { i, it ->
        BaseToken(
          id = i,
          form = it.form,
          position = Position(index = i, start = tokenPosition, end = nextPosition(it) - 1))
      }
    )
  }
}
