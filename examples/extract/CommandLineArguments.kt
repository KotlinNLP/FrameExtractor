/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package extract

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments.
 *
 * @param args the array of command line arguments
 */
class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path of the frame classifier serialized model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the file path of the frame classifier serialized model"
  )

  /**
   * The file path of the serialize model of the LHRParser.
   */
  val parserModelPath: String by parser.storing(
    "-p",
    "--parser-model-path",
    help="the file path of the serialize model of the LHRParser"
  )

  /**
   * The file path of the serialize model of the NeuralTokenizer.
   */
  val tokenizerModelPath: String by parser.storing(
    "-t",
    "--tokenizer-model-path",
    help="the file path of the serialize model of the NeuralTokenizer"
  )

  /**
   * The file path of the pre-trained word embeddings.
   */
  val embeddingsPath: String by parser.storing(
    "-e",
    "--pre-trained-word-emb-path",
    help="the file path of the pre-trained word embeddings"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String? by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  ).default { null }

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
