package com.hermesandroid.relay.wake

/**
 * A wake phrase and the exact keywords-file line the pinned KWS model
 * expects for it.
 */
data class WakeWordPhrase(
    val label: String,
    val keywordsLine: String,
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
}
