package com.hermesandroid.relay.wake

/**
 * A wake phrase and the exact keywords-file line the pinned KWS model
 * expects for it.
 */
data class WakeWordPhrase(
    val label: String,
    val keywordsLine: String,
)

/** A trained openWakeWord classifier model for one phrase (Phase 1b registers the rest). */
data class OpenWakeWordModelSpec(
    val phrase: String,
    val fileName: String,
    val sha256: String,
    val sizeBytes: Long,
    /**
     * Download base URL (the release prefix the file lives under).
     * Null → the default openWakeWord v0.5.1 release. Lets third-party models be
     * served from their own public GitHub release while keeping SHA-256 + size
     * verification identical.
     */
    val baseUrl: String? = null,
)

/**
 * Phrases offered in Settings → Voice → Wake word.
 *
 * Each [WakeWordPhrase.keywordsLine] is generated from the model's published
 * SentencePiece vocabulary (500-token uppercase BPE; `tokens.txt` pinned in
 * [WakeWordModelInstaller] at SHA-256 `fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53`).
 * Every piece was verified to exist in that vocabulary. The "Hey Hermes"
 * entry reproduces the original single-phrase keywords file byte for byte.
 */
object WakeWordPhrases {
    val SUPPORTED: List<WakeWordPhrase> = listOf(
        WakeWordPhrase("Hey Hermes", "▁HE Y ▁HER ME S @HEY_HERMES"),
        WakeWordPhrase("Hey Relay", "▁HE Y ▁RE LA Y @HEY_RELAY"),
        WakeWordPhrase("Hey Jarvis", "▁HE Y ▁JA R VI S @HEY_JARVIS"),
        WakeWordPhrase("Hey Alfred", "▁HE Y ▁A L F RE D @HEY_ALFRED"),
        WakeWordPhrase("Hey Luna", "▁HE Y ▁ LU N A @HEY_LUNA"),
    )

    val DEFAULT: WakeWordPhrase = SUPPORTED.first()

    fun byLabel(label: String): WakeWordPhrase? =
        SUPPORTED.firstOrNull { it.label.equals(label.trim(), ignoreCase = true) }

    /** Multi-line keywords file: one validated phrase per line. */
    val KEYWORDS_CONTENT: String =
        SUPPORTED.joinToString(separator = "\n") { it.keywordsLine } + "\n"

    /**
     * Trained openWakeWord classifier models. Phase 1a ships the reference "Hey Jarvis"
     * model (the only pre-trained classifier in the v0.5.1 release that matches a target
     * phrase). Phase 1b trains and registers the other four phrases here.
     *
     * "Hey Hermes" is our custom-trained model, served from its own public GitHub release
     * (baseUrl override); it keeps the exact SHA-256 + size verification of the official flow.
     */
    val OPENWAKEWORD_MODELS: List<OpenWakeWordModelSpec> = listOf(
        // "Hey Hermes" first: it is the custom-trained model and the default
        // fallback phrase when a stored phrase has no openWakeWord classifier.
        OpenWakeWordModelSpec(
            phrase = "Hey Hermes",
            fileName = "hey_hermes.tflite",
            sha256 = "6e01e6fb18be74bf39a20fba72d2977f40ac572c0e4821c24dfe24aba0755009",
            sizeBytes = 10_519_928L,
            baseUrl = "https://github.com/rodolphebertrand/hermes-relay/releases/download/wakeword-models-v1",
        ),
        OpenWakeWordModelSpec(
            phrase = "Hey Jarvis",
            fileName = "hey_jarvis_v0.1.tflite",
            sha256 = "14bff778604985e1b5c19f0f7bbe477a69cf281d8db34b232b3b972411f710e2",
            sizeBytes = 1_278_912L,
        ),
    )

    /** The trained openWakeWord model for a phrase label, or null when not yet trained. */
    fun openWakeWordSpec(label: String): OpenWakeWordModelSpec? =
        OPENWAKEWORD_MODELS.firstOrNull { it.phrase.equals(label.trim(), true) }

    /** Phrases available on the openWakeWord backend (for the UI selector). */
    val OPENWAKEWORD_AVAILABLE_LABELS: Set<String> =
        OPENWAKEWORD_MODELS.map { it.phrase }.toSet()
}
