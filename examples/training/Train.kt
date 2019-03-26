/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.Trainer
import com.kotlinnlp.frameextractor.helpers.Validator
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.xenomachina.argparser.mainBody
import buildTokensEncoderModel
import com.kotlinnlp.frameextractor.TextFrameExtractorModel
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.File
import java.io.FileInputStream

/**
 * Train a [FrameExtractorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val trainingDataset: Dataset = parsedArgs.trainingSetPath.let {
    println("Loading training dataset from '$it'...")
    Dataset.fromFile(it)
  }
  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(it)
  }

  val lssModel: LSSModel<ParsingToken, ParsingSentence> = parsedArgs.parserModelPath.let {
    println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
    LHRModel.load(FileInputStream(File(it))).lssModel
  }

  val encoderModel: TokensEncoderModel<FormToken, Sentence<FormToken>> = buildTokensEncoderModel(
    preprocessor = parsedArgs.morphoDictionaryPath.let {
      println("Loading serialized dictionary from '$it'...")
      MorphoPreprocessor(MorphologicalAnalyzer(dictionary = MorphologyDictionary.load(FileInputStream(File(it)))))
    },
    embeddingsMap = parsedArgs.embeddingsPath.let {
      println("Loading pre-trained word embeddings from '$it'...")
      EmbeddingsMap.load(it)
    },
    lssModel = lssModel)

  val extractorModel = FrameExtractorModel(
    name = parsedArgs.modelName,
    intentsConfiguration = trainingDataset.configuration,
    tokenEncodingSize = encoderModel.tokenEncodingSize,
    hiddenSize = 200)

  val model = TextFrameExtractorModel(frameExtractor = extractorModel, tokensEncoder = encoderModel)

  require(trainingDataset.configuration == validationDataset.configuration) {
    "The training dataset and the validation dataset must have the same configuration."
  }

  println()
  println("Training examples: ${trainingDataset.examples.size}.")
  println("Validation examples: ${validationDataset.examples.size}.")

  Trainer(
    model = model,
    modelFilename = parsedArgs.modelPath,
    epochs = parsedArgs.epochs,
    validator = Validator(model = model, dataset = validationDataset),
    useDropout = false
  ).train(trainingDataset)
}
