/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.helpers

import com.kotlinnlp.utils.stats.MetricCounter

/**
 * Evaluation statistics.
 *
 * @property intents a map of intents names to the related metrics
 * @property slots the metrics of the slots
 */
data class Statistics(val intents: Map<String, MetricCounter>, val slots: MetricCounter) {

  /**
   * The overall accuracy.
   */
  val accuracy: Double = (this.intents.asSequence().map { it.value.f1Score }.average() + this.slots.f1Score) / 2.0

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String {

    val maxIntentLen: Int = this.intents.keys.maxBy { it.length }!!.length + 2 // include apexes

    return """
    - Overall accuracy: ${"%.2f%%".format(100.0 * this.accuracy)}
    - Intents accuracy:
      ${this.intents.entries.joinToString("\n      ") { "%-${maxIntentLen}s : %s".format("`${it.key}`", it.value) }}
    - Slots accuracy: ${this.slots}
    """
      .removePrefix("\n")
      .trimIndent()
  }
}
