package com.hermesandroid.relay.wake

import java.io.File

/**
 * On-disk locations of the three openWakeWord TFLite models for one phrase.
 *
 * - [melspectrogram] — shared STFT/mel front-end (not phrase-specific).
 * - [embedding] — shared frozen speech embedding (not phrase-specific).
 * - [wakeWord] — per-phrase classifier (e.g. `hey_jarvis_v0.1.tflite`).
 */
data class OpenWakeWordModelFiles(
    val melspectrogram: File,
    val embedding: File,
    val wakeWord: File,
    val directory: File,
)
