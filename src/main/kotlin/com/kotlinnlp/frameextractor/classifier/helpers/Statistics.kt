/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor.classifier.helpers

/**
 * Evaluation statistics.
 *
 * @property intents statistic metric about the intents
 * @property slots statistic metric about the intent slots
 */
data class Statistics(val intents: StatsMetric, val slots: StatsMetric) {

  /**
   * Statistics for a single metric.
   *
   * @property precision precision
   * @property recall recall
   * @property f1Score the F1 score
   */
  data class StatsMetric(val precision: Double, val recall: Double, val f1Score: Double)

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String = """
    - Intents accuracy:    precision %.2f, recall %.2f, f1 score %.2f
    - Slots accuracy:      precision %.2f, recall %.2f, f1 score %.2f
    """
    .removePrefix("\n")
    .trimIndent()
    .format(
      100.0 * this.intents.precision, 100.0 * this.intents.recall, 100.0 * this.intents.f1Score,
      100.0 * this.slots.precision, 100.0 * this.slots.recall, 100.0 * this.slots.f1Score
    )
}
