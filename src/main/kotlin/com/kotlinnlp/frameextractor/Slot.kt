/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.frameextractor

/**
 * A slot of an [Intent].
 *
 * @property name the slot name
 * @property value the slot value
 */
data class Slot(val name: String, val value: String) {

  /**
   * A slot configuration.
   *
   * @property name the slot name
   * @property required whether this slot is required or not
   */
  data class Configuration(val name: String, val required: Boolean)
}