package com.hermesandroid.relay.wake

import android.content.Context

/**
 * Centralizes backend selection so the two wake-word services and the settings
 * ViewModel don't each branch on [WakeWordBackend] independently.
 *
 * - [createDetector] — builds the right [WakeWordDetector] for the configured
 *   backend + phrase, or null when the model set isn't installed.
 * - [ensureModels] — downloads + verifies the model set for the configured backend
 *   + phrase (runs on the IO dispatcher).
 */
object WakeWordEngine {

    fun isOpenWakeWord(prefs: WakeWordPreferences): Boolean =
        WakeWordBackend.fromName(prefs.backend) == WakeWordBackend.OPENWAKEWORD

    /**
     * Creates a [WakeWordDetector] for the configured backend + phrase, or null when
     * the required model set is not installed (e.g. the phrase has no trained
     * openWakeWord classifier yet, or the sherpa models are missing).
     */
    fun createDetector(context: Context, prefs: WakeWordPreferences): WakeWordDetector? {
        return if (isOpenWakeWord(prefs)) {
            val files = OpenWakeWordModelInstaller(context).installedFiles(prefs.phrase)
                ?: return null
            OpenWakeWordDetector(
                files = files,
                sensitivity = prefs.sensitivity,
                confirmationFrames = prefs.confirmationFrames,
                phrase = prefs.phrase,
            )
        } else {
            val files = WakeWordModelInstaller(context).installedFiles() ?: return null
            SherpaWakeWordDetector(
                files = files,
                sensitivity = prefs.sensitivity,
                confirmationFrames = prefs.confirmationFrames,
                phrase = prefs.phrase,
            )
        }
    }

    /**
     * Ensures the model set for the configured backend + phrase is installed.
     * The success value is the verified files (concrete type depends on backend);
     * the caller folds [Result] and ignores the value on success.
     */
    suspend fun ensureModels(context: Context, prefs: WakeWordPreferences): Result<*> {
        return if (isOpenWakeWord(prefs)) {
            OpenWakeWordModelInstaller(context).ensureInstalled(prefs.phrase)
        } else {
            WakeWordModelInstaller(context).ensureInstalled()
        }
    }
}
