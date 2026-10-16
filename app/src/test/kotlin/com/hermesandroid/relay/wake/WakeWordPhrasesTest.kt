package com.hermesandroid.relay.wake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WakeWordPhrasesTest {
    @Test
    fun catalog_containsTheFiveValidatedPhrases() {
        assertEquals(
            listOf("Hey Hermes", "Hey Relay", "Hey Jarvis", "Hey Alfred", "Hey Luna"),
            WakeWordPhrases.SUPPORTED.map { it.label },
        )
    }

    @Test
    fun default_phraseIsHeyHermesAndMatchesLegacyConstant() {
        assertEquals("Hey Hermes", WakeWordPhrases.DEFAULT.label)
        assertEquals(WakeWordPhrases.DEFAULT.label, DEFAULT_WAKE_PHRASE)
    }

    @Test
    fun byLabel_isCaseInsensitiveAndTrims() {
        assertEquals("Hey Luna", WakeWordPhrases.byLabel("hey luna")?.label)
        assertEquals("Hey Luna", WakeWordPhrases.byLabel("  HEY LUNA ")?.label)
        assertEquals("Hey Alfred", WakeWordPhrases.byLabel("Hey Alfred")?.label)
        assertNull(WakeWordPhrases.byLabel("Hey Herpes"))
        assertNull(WakeWordPhrases.byLabel("Hermes"))
        assertNull(WakeWordPhrases.byLabel("  "))
        assertNull(WakeWordPhrases.byLabel(""))
    }

    @Test
    fun keywordsContent_hasOneValidatedLinePerPhrase() {
        val lines = WakeWordPhrases.KEYWORDS_CONTENT.trimEnd().split("\n")
        assertEquals(WakeWordPhrases.SUPPORTED.size, lines.size)
        WakeWordPhrases.SUPPORTED.forEachIndexed { index, phrase ->
            assertEquals(phrase.keywordsLine, lines[index])
        }
    }

    @Test
    fun keywordsLines_followSherpaKwsLineFormat() {
        WakeWordPhrases.SUPPORTED.forEach { phrase ->
            val parts = phrase.keywordsLine.trim().split(Regex("\\s+"))
            val tag = parts.last()
            assertTrue("tag must start with '@': $tag", tag.startsWith("@"))
            assertTrue("tag must be non-empty: $tag", tag.length > 1)
            // At least one SentencePiece word-start marker (▁) in the pieces.
            assertTrue(parts.any { it.startsWith("▁") })
        }
    }

    @Test
    fun legacyHeyHermesLine_isUnchanged() {
        // Byte-for-byte backward compatibility with the original
        // single-phrase keywords file.
        assertTrue(
            WakeWordPhrases.KEYWORDS_CONTENT.startsWith("▁HE Y ▁HER ME S @HEY_HERMES\n"),
        )
    }
}
