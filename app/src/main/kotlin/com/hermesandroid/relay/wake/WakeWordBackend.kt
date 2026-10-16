package com.hermesandroid.relay.wake

/**
 * Wake-word inference backends.
 *
 * [SHERPA] — sherpa-onnx KWS (Phase 0). A single Zipformer transducer KWS model handles all
 *   five phrases (one keywords file). Already working on device.
 *
 * [OPENWAKEWORD] — openWakeWord TFLite pipeline (Phase 1a). Three lightweight models:
 *   a shared melspectrogram front-end, a shared frozen speech embedding, and a per-phrase
 *   wake-word classifier. The reference "Hey Jarvis" classifier ships pre-trained; the other
 *   four ("Hey Hermes", "Hey Relay", "Hey Alfred", "Hey Luna") are trained in Phase 1b.
 */
enum class WakeWordBackend {
    SHERPA,
    OPENWAKEWORD;

    companion object {
        /** openWakeWord is the project goal; it is the default once its models are present. */
        val DEFAULT: WakeWordBackend = OPENWAKEWORD

        fun fromName(value: String?): WakeWordBackend =
            entries.firstOrNull { it.name.equals(value, ignoreCase = true) } ?: DEFAULT
    }
}
