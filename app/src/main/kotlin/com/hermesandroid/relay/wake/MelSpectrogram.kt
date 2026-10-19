package com.hermesandroid.relay.wake

import android.content.Context
import kotlin.math.ln

/**
 * Pure-Kotlin implementation of the openWakeWord melspectrogram model
 * (melspectrogram.onnx / .tflite). Replaces the TFLite stage-1 model on
 * Android: on the S21 FE the TFLite runtime miscomputed the output shape
 * ([1, 1, frames, 32] became [N, 1, frames, 32]) and the run failed, so
 * this deterministic DSP version is used instead.
 *
 * Exact graph (verified numerically against onnxruntime, max diff 1.5e-05,
 * see MelSpectrogramTest + public/mel_test_vectors.json):
 *
 *   frames = (N - 512) / 160 + 1
 *   real[b, f]  = Σ_t wr[b, t] · x[f·160 + t]      (b = 0..256)
 *   imag[b, f]  = Σ_t wi[b, t] · x[f·160 + t]
 *   power[b, f] = real² + imag²
 *   mel[m, f]   = Σ_b power[b, f] · melW[b, m]      (m = 0..31)
 *   F = 10 · ln(max(mel, 1e-10)) / ln(10)           (= 10·log10, model raw output)
 *   out = max(F, max(frame) - 80)                    (dynamic-range floor, PER FRAME)
 *
 * The official openwakeword caller (utils.py `_streaming_melspectrogram`)
 * then applies `x / 10 + 2` before feeding the embedding model — that
 * transform is applied here when [officialTransform] is true (default).
 *
 * Weights layout (mel_weights.bin, little-endian float32, in order):
 *   wr: 257×512, wi: 257×512, melW: 257×32  → 1 085 568 bytes.
 * Audio input is RAW int16-scale float (e.g. -20000..20000), as in the
 * official pipeline (`x.astype(np.float32)` on int16 values — no /32768).
 */
class MelSpectrogram(weights: ByteArray) {
    private val winReal: FloatArray
    private val winImag: FloatArray
    private val melW: FloatArray
    private val power = FloatArray(BINS)

    init {
        require(weights.size == WEIGHT_BYTES) {
            "mel_weights.bin: expected $WEIGHT_BYTES bytes, got ${weights.size}"
        }
        fun read(offset: Int, count: Int): FloatArray {
            val b = java.nio.ByteBuffer.wrap(weights, offset, count * 4)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
            return FloatArray(count).also { b.asFloatBuffer().get(it) }
        }
        winReal = read(0, BINS * TAPS)
        winImag = read(BINS * TAPS * 4, BINS * TAPS)
        melW = read(BINS * TAPS * 8, BINS * MEL_BINS)
    }

    /**
     * Computes the melspectrogram of [samples].
     *
     * @return the number of frames produced (each frame = [MEL_BINS] floats,
     *         row-major, frame-major order: `out[frame * 32 + m]`).
     * @param out receive buffer, must hold at least `frames * 32` floats.
     * @param officialTransform apply the official `x/10 + 2` post-transform.
     */
    fun transform(samples: FloatArray, out: FloatArray, officialTransform: Boolean = true): Int {
        require(samples.size >= TAPS) { "need at least $TAPS samples, got ${samples.size}" }
        val frames = (samples.size - TAPS) / HOP + 1
        require(out.size >= frames * MEL_BINS) {
            "out too small: need ${frames * MEL_BINS}, got ${out.size}"
        }
        for (f in 0 until frames) {
            val base = f * HOP
            for (b in 0 until BINS) {
                val bo = b * TAPS
                var re = 0f
                var im = 0f
                for (t in 0 until TAPS) {
                    val s = samples[base + t]
                    re += winReal[bo + t] * s
                    im += winImag[bo + t] * s
                }
                power[b] = re * re + im * im
            }
            for (m in 0 until MEL_BINS) {
                var v = 0f
                val mOff = m
                for (b in 0 until BINS) v += power[b] * melW[b * MEL_BINS + mOff]
                out[f * MEL_BINS + m] = v
            }
        }
        // Dynamic-range floor is PER FRAME (verified against onnxruntime:
        // a global max poisons every frame — one loud frame clamps all others).
        val inv10 = 0.1f
        for (f in 0 until frames) {
            val rowStart = f * MEL_BINS
            var maxV = -Float.MAX_VALUE
            for (i in rowStart until rowStart + MEL_BINS) if (out[i] > maxV) maxV = out[i]
            var mv = maxV
            if (mv < MIN_EPS) mv = MIN_EPS
            val floor = 10f * ln(mv) / LN10 - 80f
            for (i in rowStart until rowStart + MEL_BINS) {
                var v = out[i]
                if (v < MIN_EPS) v = MIN_EPS
                v = 10f * ln(v) / LN10
                if (v < floor) v = floor
                if (officialTransform) v = v * inv10 + 2f
                out[i] = v
            }
        }
        return frames
    }

    companion object {
        /** Loads the bundled weights asset (app/src/main/assets/mel_weights.bin). */
        fun fromAsset(context: Context): MelSpectrogram {
            val bytes = context.assets.open("mel_weights.bin").use { it.readBytes() }
            return MelSpectrogram(bytes)
        }

        const val HOP = 160
        const val TAPS = 512
        const val BINS = 257
        const val MEL_BINS = 32
        const val WEIGHT_BYTES = BINS * TAPS * 2 * 4 + BINS * MEL_BINS * 4 // 1 085 568
        private const val LN10 = 2.3025851f
        private const val MIN_EPS = 1e-10f
    }
}
