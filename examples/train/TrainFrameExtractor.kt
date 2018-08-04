/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package train

import utils.buildLSSEncoder
import utils.buildSentencePreprocessor
import utils.LSSEmbeddingsEncoder
import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.Trainer
import com.kotlinnlp.frameextractor.helpers.Validator
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.frameextractor.helpers.dataset.EncodedDataset
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.keyextractors.WordKeyExtractor
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Train a [FrameExtractorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val parserModel: LHRModel = LHRModel.load(FileInputStream(File(parsedArgs.parserModelPath)))

  val embeddingsMap: EmbeddingsMapByDictionary = parsedArgs.embeddingsPath.let {
    println("Loading embeddings from '$it'...")
    EMBDLoader().load(filename = it)
  }

  val sentenceEncoder = LSSEmbeddingsEncoder(
    preprocessor = buildSentencePreprocessor(
      morphoDictionaryPath = parsedArgs.morphoDictionaryPath,
      language = parserModel.language),
    lssEncoder = parserModel.buildLSSEncoder(),
    wordEmbeddingsEncoder = EmbeddingsEncoder(
      model = EmbeddingsEncoderModel(embeddingsMap = embeddingsMap, embeddingKeyExtractor = WordKeyExtractor),
      useDropout = false))

  val trainingDataset: EncodedDataset = EncodedDataset.fromDataset(
    dataset = parsedArgs.trainingSetPath.let {
      println("Loading training dataset from '$it'...")
      Dataset.fromFile(it)
    },
    sentenceEncoder = sentenceEncoder)

  val validationDataset: EncodedDataset = EncodedDataset.fromDataset(
    dataset = parsedArgs.validationSetPath.let {
      println("Loading validation dataset from '$it'...")
      Dataset.fromFile(it)
    },
    sentenceEncoder = sentenceEncoder)

  val extractorModel = FrameExtractorModel(
    intentsConfiguration = trainingDataset.configuration,
    tokenEncodingSize = sentenceEncoder.encodingSize,
    hiddenSize = 200)

  require(trainingDataset.configuration == validationDataset.configuration) {
    "The training dataset and the validation dataset must have the same configuration."
  }

  println()
  println("Training examples: ${trainingDataset.examples.size}.")
  println("Validation examples: ${validationDataset.examples.size}.")

  Trainer(
    model = extractorModel,
    modelFilename = parsedArgs.modelPath,
    epochs = 30,
    validator = Validator(model = extractorModel, dataset = trainingDataset)
  ).train(trainingDataset)
}
