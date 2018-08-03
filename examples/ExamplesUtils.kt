/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LSSEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder.HeadsEncoder
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
 * Build an [LSSEncoder] from this LHRModel.
 *
 * @return an encoder of Latent Syntactic Structures
 */
internal fun LHRModel.buildLSSEncoder() = LSSEncoder(
  tokensEncoderWrapper = this.tokensEncoderWrapperModel.buildWrapper(useDropout = false),
  contextEncoder = ContextEncoder(this.contextEncoderModel, useDropout = false),
  headsEncoder = HeadsEncoder(this.headsEncoderModel, useDropout = false),
  virtualRoot = this.rootEmbedding.array.values)
