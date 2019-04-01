/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.frameextractor.FramesExtractorModel
import com.kotlinnlp.frameextractor.helpers.Validator
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.frameextractor.TextFramesExtractorModel
import com.kotlinnlp.frameextractor.helpers.Statistics
import com.kotlinnlp.frameextractor.helpers.dataset.Dataset
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate a [FramesExtractorModel].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val model = parsedArgs.modelPath.let {
    println("Loading text frames extractor model from '$it'...")
    TextFramesExtractorModel.load(FileInputStream(File(it)))
  }

  val validationDataset: Dataset = parsedArgs.validationSetPath.let {
    println("Loading validation dataset from '$it'...")
    Dataset.fromFile(it)
  }

  println("\nStart validation on %d examples".format(validationDataset.examples.size))

  val timer = Timer()
  val stats: Statistics = Validator(model = model, dataset = validationDataset).evaluate()

  println("Elapsed time: %s".format(timer.formatElapsedTime()))
  println()
  println("Statistics\n$stats")
}
