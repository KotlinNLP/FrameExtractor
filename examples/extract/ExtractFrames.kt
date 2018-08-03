/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package extract

import buildLSSEncoder
import buildSentencePreprocessor
import com.kotlinnlp.frameextractor.FrameExtractor
import com.kotlinnlp.frameextractor.classifier.FrameClassifier
import com.kotlinnlp.frameextractor.classifier.FrameClassifierModel
import com.kotlinnlp.frameextractor.SentenceEncoder
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.keyextractors.WordKeyExtractor
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoder
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
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
  val frameExtractor: FrameExtractor = buildFrameExtractor(parsedArgs)
  val tokenizer = NeuralTokenizer(NeuralTokenizerModel.load(FileInputStream(File(parsedArgs.tokenizerModelPath))))

  @Suppress("UNCHECKED_CAST")
  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty()) {

      break

    } else {

      tokenizer.tokenize(inputText).forEach { sentence ->

        sentence as Sentence<FormToken>

        val frame: FrameExtractor.Frame = frameExtractor.extractFrame(sentence)

        println()
        printFrame(frame = frame, sentence = sentence)
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
 * @return the frame extractor
 */
private fun buildFrameExtractor(parsedArgs: CommandLineArguments): FrameExtractor {

  val parserModel = LHRModel.load(FileInputStream(File(parsedArgs.parserModelPath)))

  val embeddingsMap: EmbeddingsMapByDictionary = parsedArgs.embeddingsPath.let {
    println("Loading embeddings from '$it'...")
    EMBDLoader().load(filename = it)
  }

  return FrameExtractor(
    classifier = FrameClassifier(model = FrameClassifierModel.load(FileInputStream(File(parsedArgs.modelPath)))),
    sentenceEncoder = SentenceEncoder(
      preprocessor = buildSentencePreprocessor(
        morphoDictionaryPath = parsedArgs.morphoDictionaryPath,
        language = parserModel.language),
      lssEncoder = parserModel.buildLSSEncoder(),
      wordEmbeddingsEncoder = EmbeddingsEncoder(
        model = EmbeddingsEncoderModel(embeddingsMap = embeddingsMap, embeddingKeyExtractor = WordKeyExtractor),
        useDropout = false)))
}

/**
 * Print a frame to the standard output.
 *
 * @param frame the frame
 * @param sentence the sentence from which the intent frame has been extracted
 */
private fun printFrame(frame: FrameExtractor.Frame, sentence: Sentence<FormToken>) {

  println("Intent: ${frame.intent.name}")

  println("Slots: %s".format(
    if (frame.intent.slots.isNotEmpty())
      frame.intent.slots.joinToString(", ") {
        "(${it.name} ${it.tokens.joinToString(" ") { sentence.tokens[it.index].form }})"
      }
    else
      "None"))

  println("Distribution:")
  frame.distribution.map.entries
    .sortedByDescending { it.value }
    .forEach { println("\t[%5.2f %%] %s".format(100.0 * it.value, it.key)) }
}
