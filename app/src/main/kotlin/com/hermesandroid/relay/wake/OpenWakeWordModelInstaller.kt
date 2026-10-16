package com.hermesandroid.relay.wake

import android.util.Log
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Downloads and verifies the openWakeWord TFLite model set for a phrase.
 *
 * Three files are involved: the shared melspectrogram front-end, the shared frozen speech
 * embedding, and the per-phrase wake-word classifier. They are downloaded to a
 * `wakeword-openwakeword/` directory, SHA-256 + size verified, and atomically promoted —
 * the same pattern as the sherpa [WakeWordModelInstaller].
 *
 * Which phrases have a trained classifier is declared in [WakeWordPhrases.openWakeWordSpec];
 * Phase 1a ships the reference "Hey Jarvis" model, Phase 1b adds the other four.
 */
class OpenWakeWordModelInstaller(context: android.content.Context) {

    private val dir: File = context.applicationContext.filesDir.resolve(DIR_NAME)

    /** True when a fully-verified model set for [phrase] is present. */
    fun isInstalled(phrase: String): Boolean = installedFiles(phrase) != null

    /**
     * Resolves the verified model files for [phrase], or null when the phrase has no
     * trained classifier yet or the files are missing/corrupt.
     */
    fun installedFiles(phrase: String): OpenWakeWordModelFiles? {
        val spec = WakeWordPhrases.openWakeWordSpec(phrase) ?: return null
        val mel = dir.resolve(MEL_FILENAME)
        val emb = dir.resolve(EMBED_FILENAME)
        val wak = dir.resolve(spec.fileName)
        if (!mel.isFile || !emb.isFile || !wak.isFile) return null
        if (mel.length() != MEL_BYTES || emb.length() != EMBED_BYTES ||
            wak.length() != spec.sizeBytes
        ) return null
        if (sha256Hex(mel) != MEL_SHA256 || sha256Hex(emb) != EMBED_SHA256 ||
            sha256Hex(wak) != spec.sha256
        ) return null
        return OpenWakeWordModelFiles(mel, emb, wak, dir)
    }

    /**
     * Downloads + verifies the model set for [phrase] (on the IO dispatcher).
     * Returns [Ok] with the verified files, or [failure] with a reason.
     */
    suspend fun ensureInstalled(phrase: String): Result<OpenWakeWordModelFiles> =
        withContext(Dispatchers.IO) {
            runCatching {
                installedFiles(phrase)?.let { return@runCatching it }
                val spec = WakeWordPhrases.openWakeWordSpec(phrase)
                    ?: throw IOException("No openWakeWord model trained for phrase '$phrase' yet")
                val staging = dir.resolveSibling(DIR_NAME + ".tmp")
                if (staging.exists()) staging.deleteRecursively()
                staging.mkdirs()
                download(MEL_FILENAME, MEL_SHA256, MEL_BYTES, staging)
                download(EMBED_FILENAME, EMBED_SHA256, EMBED_BYTES, staging)
                download(spec.fileName, spec.sha256, spec.sizeBytes, staging)
                promote(staging)
                installedFiles(phrase) ?: throw IOException("openWakeWord install failed to verify")
            }
        }

    private fun download(filename: String, sha256: String, expectedBytes: Long, staging: File) {
        val target = staging.resolve(filename)
        if (target.isFile && sha256Hex(target) == sha256 && target.length() == expectedBytes) {
            return
        }
        target.delete()
        val url = "$BASE_URL/$filename"
        Log.i(TAG, "downloading $url")
        var conn: HttpURLConnection? = null
        try {
            conn = URL(url).openConnection() as HttpURLConnection
            conn.connectTimeout = 20_000
            conn.readTimeout = 60_000
            val inStream = BufferedInputStream(conn.inputStream)
            val outStream = BufferedOutputStream(FileOutputStream(target))
            val buf = ByteArray(64 * 1024)
            var n = inStream.read(buf)
            while (n >= 0) {
                outStream.write(buf, 0, n)
                n = inStream.read(buf)
            }
            outStream.flush()
            inStream.close()
            outStream.close()
        } finally {
            conn?.disconnect()
        }
        if (sha256Hex(target) != sha256) {
            target.delete()
            throw IOException("SHA-256 mismatch for $filename")
        }
        if (target.length() != expectedBytes) {
            target.delete()
            throw IOException("size mismatch for $filename: ${target.length()} != $expectedBytes")
        }
    }

    private fun promote(staging: File) {
        if (dir.exists()) dir.deleteRecursively()
        if (!staging.renameTo(dir)) {
            staging.copyRecursively(dir, overwrite = true)
            staging.deleteRecursively()
        }
        dir.resolve("openwakeword-verified").writeText("verified\n")
    }

    /** SHA-256 hex of a file (local impl to avoid a new util dependency). */
    private fun sha256Hex(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(DEFAULT_BUFFER_SIZE)
            var n = input.read(buf)
            while (n >= 0) {
                digest.update(buf, 0, n)
                n = input.read(buf)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "OpenWakeWordInstaller"
        const val DIR_NAME = "wakeword-openwakeword"
        private const val BASE_URL =
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

        // Shared, phrase-independent models (SHA-256 + size from the v0.5.1 release).
        private const val MEL_FILENAME = "melspectrogram.tflite"
        private const val MEL_SHA256 =
            "93a22705928f35f080560937a71483c498c8d489195b178e43112832248a32d7"
        private const val MEL_BYTES = 1_147_076L

        private const val EMBED_FILENAME = "embedding_model.tflite"
        private const val EMBED_SHA256 =
            "c0915d701d92f6f32651d5621100d712240e60d76a8a2c14610f15841b01d130"
        private const val EMBED_BYTES = 1_302_492L
    }
}
