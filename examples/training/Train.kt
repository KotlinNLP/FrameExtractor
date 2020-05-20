/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.frameextractor.FramesExtractorModel
import com.kotlinnlp.frameextractor.helpers.Trainer
import com.kotlinnlp.frameextractor.helpers.Validator
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.frameextractor.TextFramesExtractorModel
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * Train a [FramesExtractorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)
  val optimizeEmbeddings: Boolean = !parsedArgs.noEmbeddingsOptimization

  val trainingDataset: Dataset = parsedArgs.trainingSetPath.let {
    println("Loading training dataset from '$it'...")
    Dataset.fromFile(it)
  }
  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(it)
  }

  val preTrainedEmbeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }
  val emptyEmbeddingsMap = EmbeddingsMap<String>(size = 100)
  val encoderModel: TokensEncoderModel<FormToken, Sentence<FormToken>> = buildTokensEncoderModel(
    preTrainedEmbeddingsMap = preTrainedEmbeddingsMap,
    emptyEmbeddingsMap = emptyEmbeddingsMap,
    optimizeEmbeddings = optimizeEmbeddings)
  val extractorModel = FramesExtractorModel(
    name = parsedArgs.modelName,
    intentsConfiguration = trainingDataset.configuration,
    tokenEncodingSize = encoderModel.tokenEncodingSize,
    hiddenSize = 200)

  val model = TextFramesExtractorModel(frameExtractor = extractorModel, tokensEncoder = encoderModel)

  require(trainingDataset.configuration == validationDataset.configuration) {
    "The training dataset and the validation dataset must have the same configuration."
  }

  if (optimizeEmbeddings) preTrainedEmbeddingsMap.addAll(trainingDataset.examples.map { it.sentence })
  emptyEmbeddingsMap.addAll(trainingDataset.examples.map { it.sentence })

  println()
  println("Training examples: ${trainingDataset.examples.size}.")
  println("Validation examples: ${validationDataset.examples.size}.")

  Trainer(
    model = model,
    modelFilename = parsedArgs.modelPath,
    epochs = parsedArgs.epochs,
    encoderUpdateMethod = AdaGradMethod(learningRate = 0.1),
    extractorUpdateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    validator = Validator(model = model, dataset = validationDataset),
    useDropout = false
  ).train(trainingDataset)
}
