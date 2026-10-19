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
     * Resolves a previously installed custom classifier (see [ensureCustomInstalled])
     * for the exact [url], or null when anything is missing/corrupt or the cached
     * model was downloaded from a different URL.
     */
    fun installedCustomFiles(url: String, sha256: String?): OpenWakeWordModelFiles? {
        val cleanUrl = url.trim()
        if (cleanUrl.isEmpty()) return null
        val mel = dir.resolve(MEL_FILENAME)
        val emb = dir.resolve(EMBED_FILENAME)
        val wak = dir.resolve(CUSTOM_FILENAME)
        if (!mel.isFile || !emb.isFile || !wak.isFile) return null
        if (mel.length() != MEL_BYTES || emb.length() != EMBED_BYTES) return null
        if (sha256Hex(mel) != MEL_SHA256 || sha256Hex(emb) != EMBED_SHA256) return null
        if (wak.length() < MIN_CUSTOM_BYTES) return null
        val expected = sha256?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
        if (expected != null && sha256Hex(wak) != expected) return null
        // The meta file records which URL produced this classifier, so switching
        // URLs always forces a fresh download instead of reusing the wrong model.
        val meta = dir.resolve(CUSTOM_META_FILENAME)
        if (!meta.isFile) return null
        val metaUrl = meta.readText().lineSequence()
            .firstOrNull { it.startsWith("url=") }
            ?.removePrefix("url=")
        if (metaUrl != cleanUrl) return null
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
                val staging = prepareStaging()
                download(MEL_FILENAME, MEL_SHA256, MEL_BYTES, staging)
                download(EMBED_FILENAME, EMBED_SHA256, EMBED_BYTES, staging)
                download(spec.fileName, spec.sha256, spec.sizeBytes, staging, spec.baseUrl ?: BASE_URL)
                promote(staging)
                installedFiles(phrase) ?: throw IOException("openWakeWord install failed to verify")
            }
        }

    /**
     * Downloads + verifies a custom classifier from a full [url] (any HTTP(S) server,
     * e.g. a private LAN host) plus the shared melspectrogram/embedding models, then
     * atomically promotes the set. [sha256] (optional) is used to verify the download;
     * without it, a minimum-size sanity check guards against error pages.
     */
    suspend fun ensureCustomInstalled(url: String, sha256: String?): Result<OpenWakeWordModelFiles> =
        withContext(Dispatchers.IO) {
            runCatching {
                val cleanUrl = url.trim()
                require(cleanUrl.isNotEmpty()) { "custom model URL is empty" }
                val expected = sha256?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
                require(expected == null || (expected.length == 64 && expected.all { it.isDigit() || it in 'a'..'f' })) {
                    "SHA-256 invalide : ${expected?.length} caractères (attendu 64 hex)"
                }
                installedCustomFiles(cleanUrl, expected)?.let { return@runCatching it }
                val staging = prepareStaging()
                download(MEL_FILENAME, MEL_SHA256, MEL_BYTES, staging)
                download(EMBED_FILENAME, EMBED_SHA256, EMBED_BYTES, staging)
                staging.resolve(CUSTOM_META_FILENAME).writeText(
                    buildString {
                        appendLine("url=$cleanUrl")
                        if (expected != null) appendLine("sha256=$expected")
                    },
                )
                downloadCustom(cleanUrl, expected, staging)
                promote(staging)
                installedCustomFiles(cleanUrl, expected)
                    ?: throw IOException("custom wake model install failed verification")
            }
        }

    /**
     * Fresh staging area that preserves whatever is already installed (other phrase
     * classifiers, a previous custom model, the verified marker) so [promote] never
     * wipes out models that are still in use.
     */
    private fun prepareStaging(): File {
        val staging = dir.resolveSibling(DIR_NAME + ".tmp")
        if (staging.exists()) staging.deleteRecursively()
        staging.mkdirs()
        if (dir.exists()) dir.copyRecursively(staging, overwrite = true)
        return staging
    }

    private fun downloadCustom(url: String, expectedSha: String?, staging: File) {
        val target = staging.resolve(CUSTOM_FILENAME)
        // Reuse a cached copy only when it was downloaded for the SAME url:
        // prepareStaging() copies the previously-installed custom model into the
        // staging area, and without this url check an url change without a sha
        // would silently re-promote the OLD model under the NEW url's meta.
        val stagingMeta = staging.resolve(CUSTOM_META_FILENAME)
        val stagingMetaUrl = if (stagingMeta.isFile) stagingMeta.readText().lineSequence()
            .firstOrNull { it.startsWith(\"url=\") }?.removePrefix(\"url=\") else null
        if (target.isFile && target.length() >= MIN_CUSTOM_BYTES && stagingMetaUrl == url &&
            (expectedSha == null || sha256Hex(target) == expectedSha)
        ) {
            return
        }
        target.delete()
        Log.i(TAG, "downloading custom wake model from $url")
        var conn: HttpURLConnection? = null
        try {
            conn = URL(url).openConnection() as HttpURLConnection
            conn.connectTimeout = 20_000
            conn.readTimeout = 120_000
            if (conn.responseCode != HttpURLConnection.HTTP_OK) {
                throw IOException("HTTP ${conn.responseCode} for custom model URL")
            }
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
        if (target.length() < MIN_CUSTOM_BYTES) {
            target.delete()
            throw IOException(
                "custom model too small (${target.length()} bytes) — likely an error page, not a model",
            )
        }
        if (expectedSha != null && sha256Hex(target) != expectedSha) {
            target.delete()
            throw IOException("SHA-256 mismatch for custom wake model")
        }
    }

    private fun download(
        filename: String,
        sha256: String,
        expectedBytes: Long,
        staging: File,
        baseUrl: String = BASE_URL,
    ) {
        val target = staging.resolve(filename)
        if (target.isFile && sha256Hex(target) == sha256 && target.length() == expectedBytes) {
            return
        }
        target.delete()
        val url = "$baseUrl/$filename"
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
            "96fa0adccb6e8cf95cb14465409a1a2898ee4a96a85bb9ed3c7eb0e68bf163e8"
        private const val MEL_BYTES = 1_092_516L

        private const val EMBED_FILENAME = "embedding_model.tflite"
        private const val EMBED_SHA256 =
            "c0aea21eb84a4ce90a08c870da41b7a7173b45269e6a3207c71d67c40f3a59d8"
        private const val EMBED_BYTES = 1_330_312L

        // Custom classifier (user-supplied URL): filename is fixed, size is not
        // known in advance — a minimum-size floor catches HTML error pages.
        const val CUSTOM_FILENAME = "custom_wakeword.tflite"
        const val CUSTOM_META_FILENAME = "custom_wakeword.meta"
        const val MIN_CUSTOM_BYTES = 100_000L
    }
}
