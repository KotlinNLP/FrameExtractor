/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.Validator
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.frameextractor.helpers.dataset.EncodedDataset
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.xenomachina.argparser.mainBody
import buildTokensEncoder
import com.kotlinnlp.frameextractor.helpers.Statistics
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate a [FrameExtractorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val lssModel: LSSModel<ParsingToken, ParsingSentence> = parsedArgs.parserModelPath.let {
    println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
    LHRModel.load(FileInputStream(File(it))).lssModel
  }

  val tokensEncoder: TokensEncoder<FormToken, Sentence<FormToken>> = buildTokensEncoder(
    preprocessor = parsedArgs.morphoDictionaryPath.let {
      println("Loading serialized dictionary from '$it'...")
      MorphoPreprocessor(MorphologicalAnalyzer(
        language = lssModel.language,
        dictionary = MorphologyDictionary.load(FileInputStream(File(it)))))
    },
    embeddingsMap = parsedArgs.embeddingsPath.let {
      println("Loading pre-trained word embeddings from '$it'...")
      EmbeddingsMap.load(it)
    },
    lssModel = lssModel)

  val validationDataset: EncodedDataset = EncodedDataset.fromDataset(
    dataset = parsedArgs.validationSetPath.let {
      println("Loading validation dataset from '$it'...")
      Dataset.fromFile(it)
    },
    tokensEncoder = tokensEncoder)

  val extractorModel = parsedArgs.modelPath.let {
    println("Loading frame extractor model from '$it'...")
    FrameExtractorModel.load(FileInputStream(File(it)))
  }

  println("\nStart validation on %d examples".format(validationDataset.examples.size))

  val timer = Timer()
  val stats: Statistics = Validator(model = extractorModel, dataset = validationDataset).evaluate()
  val accuracy: Double = stats.intents.f1Score * stats.slots.f1Score

  println("Elapsed time: %s".format(timer.formatElapsedTime()))

  println("\nAccuracy: %.2f%%".format(100.0 * accuracy))
  println("\nStatistics\n$stats")
}
