/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package extract

import buildTokensEncoder
import com.kotlinnlp.frameextractor.FrameExtractor
import com.kotlinnlp.frameextractor.FrameExtractorModel
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Extract frames from a text.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)
  val tokenizer = NeuralTokenizer(NeuralTokenizerModel.load(FileInputStream(File(parsedArgs.tokenizerModelPath))))
  val textFramesExtractor: TextFramesExtractor = buildTextFramesExtractor(parsedArgs)

  @Suppress("UNCHECKED_CAST")
  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty()) {

      break

    } else {

      tokenizer.tokenize(inputText).forEach { sentence ->

        sentence as Sentence<FormToken>

        println()
        textFramesExtractor.extractFrames(sentence).print(sentence)
      }
    }
  }

  println("\nExiting...")
}

/**
 * Read a value from the standard input.
 *
 * @return the string read
 */
private fun readValue(): String {

  print("\nExtract frames from a text (empty to exit): ")

  return readLine()!!
}

/**
 * @param parsedArgs the command line parsed arguments
 *
 * @return the text frames extractor
 */
private fun buildTextFramesExtractor(parsedArgs: CommandLineArguments): TextFramesExtractor {

  val lssModel: LSSModel<ParsingToken, ParsingSentence> = parsedArgs.parserModelPath.let {
    println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
    LHRModel.load(FileInputStream(File(it))).lssModel
  }

  val tokensEncoder = buildTokensEncoder(
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

  val model: FrameExtractorModel = parsedArgs.modelPath.let {
    println("Loading frame extractor model from '$it'...")
    FrameExtractorModel.load(FileInputStream(File(it)))
  }

  println("\nFrame Extractor model: ${model.name}")

  return TextFramesExtractor(extractor = FrameExtractor(model), tokensEncoder = tokensEncoder)
}

/**
 * Print this frame to the standard output.
 *
 * @param sentence the sentence from which this frame has been extracted
 */
private fun TextFramesExtractor.Frame.print(sentence: Sentence<FormToken>) {

  println("Intent: ${this.intent.name}")

  println("Slots: %s".format(
    if (this.intent.slots.isNotEmpty())
      this.intent.slots.joinToString(", ") { slot ->
        "(${slot.name} ${slot.tokens.joinToString(" ") { sentence.tokens[it.index].form }})"
      }
    else
      "None"))

  println("Distribution:")
  this.distribution.map.entries
    .sortedByDescending { it.value }
    .forEach { println("\t[%5.2f %%] %s".format(100.0 * it.value, it.key)) }
}
