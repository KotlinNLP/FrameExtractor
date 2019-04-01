/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation.separatemodels

import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.frameextractor.helpers.Validator
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.frameextractor.TextFramesExtractorModel
import com.kotlinnlp.frameextractor.helpers.Statistics
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate a [FrameExtractorModel] with a transient embeddings encoder, loading the embeddings separately.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val model = parsedArgs.modelPath.let {
    println("Loading text frames extractor model from '$it'...")
    TextFramesExtractorModel.load(FileInputStream(File(it)))
  }

  val embeddingsMap: EmbeddingsMap<String> = parsedArgs.embeddingsPath.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(it)
  }

  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(it)
  }

  val firstEncoder: TokensEncoderWrapperModel<*, *, *, *> =
    (model.tokensEncoder as EnsembleTokensEncoderModel)
      .components.first().model as TokensEncoderWrapperModel<*, *, *, *>

  (firstEncoder.model as EmbeddingsEncoderModel.Transient).setEmbeddingsMap(embeddingsMap)

  println("\nStart validation on %d examples".format(validationDataset.examples.size))

  val timer = Timer()
  val stats: Statistics = Validator(model = model, dataset = validationDataset).evaluate()

  println("Elapsed time: %s".format(timer.formatElapsedTime()))
  println()
  println("Statistics\n$stats")
}
