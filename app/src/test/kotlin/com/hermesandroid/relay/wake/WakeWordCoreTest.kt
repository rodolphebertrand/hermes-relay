package com.hermesandroid.relay.wake

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class WakeWordCoreTest {
    @Before
    fun setUp() {
        MicrophoneOwnershipCoordinator.resetForTest()
        WakeWordActivationCoordinator.resetForTest()
    }

    @After
    fun tearDown() {
        MicrophoneOwnershipCoordinator.resetForTest()
        WakeWordActivationCoordinator.resetForTest()
    }

    @Test
    fun microphoneLease_isExclusiveAndRejectsStaleRelease() {
        val wakeLease = MicrophoneOwnershipCoordinator.tryAcquire(MicrophoneOwner.WakeWord)
        requireNotNull(wakeLease)
        assertEquals(MicrophoneOwner.WakeWord, MicrophoneOwnershipCoordinator.owner.value)
        assertNull(MicrophoneOwnershipCoordinator.tryAcquire(MicrophoneOwner.VoiceCapture))

        val stale = MicrophoneLease(MicrophoneOwner.WakeWord, "stale")
        assertFalse(MicrophoneOwnershipCoordinator.release(stale))
        assertEquals(MicrophoneOwner.WakeWord, MicrophoneOwnershipCoordinator.owner.value)

        assertTrue(MicrophoneOwnershipCoordinator.release(wakeLease))
        assertNull(MicrophoneOwnershipCoordinator.owner.value)
        assertEquals(
            MicrophoneOwner.VoiceCapture,
            MicrophoneOwnershipCoordinator.tryAcquire(MicrophoneOwner.VoiceCapture)?.owner,
        )
    }

    @Test
    fun activation_remainsPendingUntilMatchingConsumer() {
        val activation = WakeWordActivation(
            id = "activation-1",
            startNewSession = true,
            profileRouting = WakeWordProfileRouting(),
            source = WakeWordActivationSource.SystemAssistant,
        )
        WakeWordActivationCoordinator.request(activation)

        assertEquals(activation, WakeWordActivationCoordinator.pending.value)
        assertEquals(
            WakeWordActivationSource.SystemAssistant,
            WakeWordActivationCoordinator.pending.value?.source,
        )
        assertFalse(WakeWordActivationCoordinator.consume("other"))
        assertEquals(activation, WakeWordActivationCoordinator.pending.value)
        assertTrue(WakeWordActivationCoordinator.consume("activation-1"))
        assertNull(WakeWordActivationCoordinator.pending.value)
    }

    @Test
    fun confirmationFrames_mapToSherpaTrailingBlanksAndClamp() {
        assertEquals(1, WakeWordTuning.trailingBlanks(-1))
        assertEquals(3, WakeWordTuning.trailingBlanks(3))
        assertEquals(5, WakeWordTuning.trailingBlanks(9))
    }

    @Test
    fun completedKeyword_matchesConfiguredPhraseOnly() {
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("HEY_HERMES", "Hey Hermes"))
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("  Hey   Hermes ", "Hey Hermes"))
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("HEY_LUNA", "Hey Luna"))
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("hey_jarvis", "HEY JARVIS"))
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("HEY_ALFRED", "Hey Alfred"))
        assertTrue(WakeWordTuning.matchesConfiguredPhrase("HEY_RELAY", "Hey Relay"))
        // Other phrases must not activate the configured one.
        assertFalse(WakeWordTuning.matchesConfiguredPhrase("HEY_HERMES", "Hey Luna"))
        assertFalse(WakeWordTuning.matchesConfiguredPhrase("HEY_JARVIS", "Hey Alfred"))
        assertFalse(WakeWordTuning.matchesConfiguredPhrase("Hey Herpes", "Hey Hermes"))
        assertFalse(WakeWordTuning.matchesConfiguredPhrase("", "Hey Hermes"))
    }

    @Test
    fun sensitivityThreshold_higherMeansStricterAndClamps() {
        assertTrue(WakeWordTuning.threshold(0.8f) > WakeWordTuning.threshold(0.3f))
        assertEquals(0.2f, WakeWordTuning.threshold(-1f))
        assertEquals(0.9f, WakeWordTuning.threshold(2f))
    }

    @Test
    fun inputLevel_reportsPeakAndHandlesEmptyFrames() {
        assertEquals(0f, wakeWordInputLevel(shortArrayOf(), 0))
        assertEquals(0.5f, wakeWordInputLevel(shortArrayOf(0, 16_384, -4_000), 3))
        assertEquals(1f, wakeWordInputLevel(shortArrayOf(Short.MIN_VALUE), 1))
    }
}
