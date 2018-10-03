/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.keyextractors.WordKeyExtractor
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoder
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.lss.LSSTokensEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import java.io.File
import java.io.FileInputStream

/**
 * Build a [SentencePreprocessor].
 *
 * @param morphoDictionaryPath the path of the serialized morphology dictionary
 * @param language the language in which to process the sentence
 *
 * @return a new sentence preprocessor
 */
internal fun buildSentencePreprocessor(morphoDictionaryPath: String?, language: Language): SentencePreprocessor {

  return morphoDictionaryPath?.let {

    println("Loading serialized dictionary from '$it'...")

    MorphoPreprocessor(
      MorphologicalAnalyzer(language = language, dictionary = MorphologyDictionary.load(FileInputStream(File(it))))
    )

  } ?: BasePreprocessor()
}

/**
 * Build an [EnsembleTokensEncoder] composed by an embeddings encoder and an LSS encoder.
 *
 * @param preprocessor a sentence preprocessor
 * @param embeddingsMap an embeddings map by dictionary
 * @param lssModel the model of an LSS encoder
 *
 * @return a new tokens encoder
 */
internal fun buildTokensEncoder(preprocessor: SentencePreprocessor,
                                embeddingsMap: EmbeddingsMapByDictionary,
                                lssModel: LSSModel<ParsingToken, ParsingSentence>) = EnsembleTokensEncoder(
  model = EnsembleTokensEncoderModel(
    models = listOf(
      TokensEncoderWrapperModel(
        model = EmbeddingsEncoderModel(embeddingsMap = embeddingsMap, embeddingKeyExtractor = WordKeyExtractor()),
        converter = FormSentenceConverter(preprocessor)),
      TokensEncoderWrapperModel(
        model = LSSTokensEncoderModel(lssModel = lssModel),
        converter = FormSentenceConverter(preprocessor)))
  ),
  useDropout = false)

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
  override fun convert(sentence: Sentence<FormToken>): ParsingSentence {

    var tokenPosition = 0

    fun nextPosition(token: FormToken): Int {
      tokenPosition += token.form.length + 1
      return tokenPosition
    }

    return ParsingSentence(
      tokens = sentence.tokens.mapIndexed { i, it ->
        ParsingToken(
          id = i,
          form = it.form,
          position = Position(index = i, start = tokenPosition, end = nextPosition(it) - 1),
          morphologies = emptyList())
      }
    )
  }
}
