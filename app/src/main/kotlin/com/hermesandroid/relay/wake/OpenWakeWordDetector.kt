package com.hermesandroid.relay.wake

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter

/** Number of 30 ms embedding frames the openWakeWord classifier expects. */
private const val OWW_WAKE_FRAMES = 16
/** Dimensionality of each openWakeWord embedding. */
private const val OWW_EMBED_DIM = 96

/**
 * Fills a flat openWakeWord wake-classifier input buffer with the last
 * [OWW_WAKE_FRAMES] embeddings of [embeddings].
 *
 * openWakeWord ships classifier inputs in two memory layouts:
 *   - `transposed = false`: [frames, dim] (16×96) — the official release models;
 *   - `transposed = true`:  [dim, frames] (96×16) — models exported via the official
 *     training pipeline (e.g. a custom model trained on a specific voice).
 *
 * The caller must pick the layout from the classifier's declared input shape
 * (see [OpenWakeWordDetector] init). Pure function, dependency-free, so it can
 * be unit-tested on the JVM.
 */
fun fillWakeInput(embeddings: List<FloatArray>, out: FloatArray, transposed: Boolean) {
    require(embeddings.size >= OWW_WAKE_FRAMES) { "need at least $OWW_WAKE_FRAMES embeddings" }
    require(out.size == OWW_WAKE_FRAMES * OWW_EMBED_DIM) {
        "expected ${OWW_WAKE_FRAMES * OWW_EMBED_DIM} floats"
    }
    val start = embeddings.size - OWW_WAKE_FRAMES
    if (transposed) {
        var w = 0
        for (d in 0 until OWW_EMBED_DIM) {
            for (k in 0 until OWW_WAKE_FRAMES) {
                out[w++] = embeddings[start + k][d]
            }
        }
    } else {
        var w = 0
        for (k in 0 until OWW_WAKE_FRAMES) {
            embeddings[start + k].copyInto(out, w, 0, OWW_EMBED_DIM)
            w += OWW_EMBED_DIM
        }
    }
}

/**
 * openWakeWord wake-word detector ([WakeWordDetector] implementation).
 *
 * Pipeline (faithful to `openwakeword.utils.AudioPreprocessor`):
 *   1. Audio is buffered into 1280-sample (80 ms @ 16 kHz) chunks.
 *   2. Each chunk is run through the melspectrogram with 480 samples of left
 *      context (pure-Kotlin [MelSpectrogram] — see below); the new mel frames are
 *      appended to a rolling buffer (kept at <= 76 frames).
 *   3. The post-mel transform `spec = spec / 10 + 2` is applied INSIDE
 *      [MelSpectrogram] (MANDATORY — matches openwakeword/utils.py) before the
 *      embedding model.
 *   4. The last 76 mel frames form the window for the embedding model -> one 96-dim
 *      embedding, appended to a rolling buffer (maxlen 120).
 *   5. The wake-word classifier consumes the LAST 16 embeddings -> score 0..1. The
 *      classifier input layout (16×96 or 96×16) is auto-detected from the model's
 *      declared input shape at init (see [fillWakeInput]).
 *   6. A detection fires when `score >= threshold` for `confirmationFrames` consecutive
 *      80 ms windows (debounce against single-frame spikes).
 *
 * MEL SPECTROGRAM: the TFLite melspectrogram model broke on-device (wrong output
 * shape [N,1,frames,32] instead of [1,1,frames,32]) — root cause: the old code fed
 * TFLite a FLAT rank-1 array, and TFLite Java resizes the input tensor to the shape
 * of the Java array it receives (see the wrapper comment near embInArr). The mel is
 * now computed in pure Kotlin ([MelSpectrogram]), numerically identical to the ONNX
 * model (validated against onnxruntime, max diff 1.5e-05). Only the embedding +
 * classifier models still run through TFLite — always with N-D arrays matching
 * their declared shapes.
 *
 * INPUT SCALING: openWakeWord casts int16 -> float32 WITHOUT dividing by 32768 (values stay
 * in [-32768, 32767]) — unlike sherpa-onnx, which normalizes. This is critical for accuracy.
 *
 * WARMUP: the first ~10 chunks (≈800 ms) fill the 76-frame mel window and 16-embedding
 * window, so early scores are 0 until the buffers are populated (expected).
 */
class OpenWakeWordDetector(
    files: OpenWakeWordModelFiles,
    sensitivity: Float,
    confirmationFrames: Int,
    phrase: String,
    private val mel: MelSpectrogram,
) : WakeWordDetector {

    private val options = Interpreter.Options().apply {
        numThreads = 2
        // CPU inference: reliable and fast enough for the 80 ms cadence. A LiteRT GPU
        // delegate can be layered on later as a measured optimization.
    }

    private val embedding = Interpreter(files.embedding, options)
    private val wakeWord = Interpreter(files.wakeWord, options)

    // Output buffers, sized against the actual (post-resize) tensor shapes.
    private val melOut = FloatArray(MAX_MEL_FRAMES * MEL_BINS)
    private val embOut = FloatArray(EMBED_DIM)
    private val wakeOut = FloatArray(1)

    // TFLite Java resizes a tensor to the shape of the array handed to run(): a flat
    // FloatArray becomes a rank-1 tensor and breaks the graph — the embedding model's
    // PAD node then fails with "paddings 4 != dims 1" (2.16.1, caught) and the runtime
    // SIGSEGV'd outright (2.17.0); the old melspectrogram showed the same disease as a
    // wrong output shape. run() must therefore receive arrays matching the DECLARED
    // tensor shapes exactly — the same N-D layout the official openwakeword Python
    // feeds (numpy [1,76,32,1]). The flat buffers above stay the fill logic; these
    // wrappers only give the tensors their true shape.
    private val embInArr = Array(1) { Array(MEL_FRAMES) { Array(MEL_BINS) { FloatArray(1) } } }
    private val embOutArr = Array(1) { Array(1) { Array(1) { FloatArray(EMBED_DIM) } } }
    private val wakeInArr = Array(1) { Array(WAKE_FRAMES) { FloatArray(EMBED_DIM) } }
    private val wakeInArrT = Array(1) { Array(EMBED_DIM) { FloatArray(WAKE_FRAMES) } }
    private val wakeOutArr = arrayOf(FloatArray(1))

    // Layout of the wake-classifier input, decided at init from the model's declared
    // shape (true for 96×16 "dim×frames" exports, false for 16×96 "frames×dim").
    private var wakeInputIsTransposed = false

    init {
        // Pin the embedding model input to the fixed 76×32 window (1,76,32,1) so the
        // runtime never has to infer a dynamic shape.
        runCatching {
            embedding.resizeInput(0, intArrayOf(1, MEL_FRAMES, MEL_BINS, 1), true)
        }.onFailure { Log.w(TAG, "embedding resizeInput: $it") }
        embedding.allocateTensors()
        // Pin the classifier input to its declared frame order once the layout is known.
        val wakeShape = wakeWord.getInputTensor(0).shape()
        wakeInputIsTransposed = wakeShape.size == 3 &&
            wakeShape[1] == EMBED_DIM && wakeShape[2] == WAKE_FRAMES
        Log.i(
            TAG,
            "wake classifier input ${wakeShape.contentToString()}, output " +
                "${wakeWord.getOutputTensor(0).shape().contentToString()} -> " +
                if (wakeInputIsTransposed) "dim×frames layout (96×16)" else "frames×dim layout (16×96)",
        )
        // Pin the classifier input to the exact layout we feed (no dynamic shapes).
        runCatching {
            wakeWord.resizeInput(
                0,
                if (wakeInputIsTransposed) intArrayOf(1, EMBED_DIM, WAKE_FRAMES)
                else intArrayOf(1, WAKE_FRAMES, EMBED_DIM),
                true,
            )
        }.onFailure { Log.w(TAG, "wake resizeInput: $it") }
        wakeWord.allocateTensors()
        val embInShape = embedding.getInputTensor(0).shape()
        Log.i(TAG, "embedding input ${embInShape.contentToString()}, output ${embedding.getOutputTensor(0).shape().contentToString()}")
    }

    // --- Streaming buffers (mirror openwakeword.utils.AudioPreprocessor) ---
    private val melIn = FloatArray(MEL_INPUT_SAMPLES)
    private var history = ShortArray(0)              // last MEL_CONTEXT samples processed
    private val melFrames = ArrayDeque<FloatArray>() // rolling 76 frames of 32-dim
    private val embWindow = FloatArray(MEL_FRAMES * MEL_BINS)
    private val embeddings = ArrayDeque<FloatArray>() // rolling 120 embeddings of 96-dim
    private val wakeIn = FloatArray(WAKE_FRAMES * EMBED_DIM)
    private var remainder = ShortArray(0)            // < 1280 leftover samples
    private var confirmHits = 0
    private var scoreLogChunks = 0
    private var closed = false

    private val threshold: Float = thresholdFor(sensitivity)
    private val confirmationsNeeded: Int = confirmationFrames.coerceIn(1, 8)

    override fun accept(samples: ShortArray, count: Int): Boolean {
        if (count <= 0) return false
        // Combine the leftover remainder with the new samples.
        val total = remainder.size + count
        val all = ShortArray(total)
        System.arraycopy(remainder, 0, all, 0, remainder.size)
        System.arraycopy(samples, 0, all, remainder.size, count)
        remainder = ShortArray(0)
        if (total < CHUNK_SAMPLES) {
            remainder = all
            return false
        }
        val emit = total - (total % CHUNK_SAMPLES)
        val emitted = all.copyOfRange(0, emit)
        remainder = all.copyOfRange(emit, total)
        for (c in 0 until emit / CHUNK_SAMPLES) {
            processChunk(emitted.copyOfRange(c * CHUNK_SAMPLES, (c + 1) * CHUNK_SAMPLES))
        }
        // Diagnostic: surface the raw classifier score so false positives can be
        // attributed (model calibration vs broken features) straight from logcat.
        // Any score >= 0.25 is logged; otherwise one heartbeat every ~125 chunks (~10 s).
        val score = currentScore()
        if (score >= 0.25f || ++scoreLogChunks >= 125) {
            scoreLogChunks = 0
            Log.i(TAG, "wake score=$score threshold=$threshold emb=${embeddings.size}")
        }
        return score >= threshold && takeConfirmation()
    }

    private fun processChunk(chunk: ShortArray) {
        // Build the melspectrogram input: [480 left context (zero-padded)] + [1280 chunk].
        for (i in 0 until MEL_CONTEXT) melIn[i] = 0f
        val ctx = history.size.coerceAtMost(MEL_CONTEXT)
        for (i in 0 until ctx) melIn[MEL_CONTEXT - ctx + i] = history[history.size - ctx + i].toFloat()
        for (i in chunk.indices) melIn[MEL_CONTEXT + i] = chunk[i].toFloat()
        // Pure-Kotlin melspectrogram; includes the mandatory `x/10 + 2` post-transform.
        val frames = mel.transform(melIn, melOut)
        val newFrames = minOf(frames, MEL_FRAMES_PER_CHUNK)
        for (f in (frames - newFrames) until frames) {
            val frame = FloatArray(MEL_BINS)
            melOut.copyInto(frame, 0, f * MEL_BINS, (f + 1) * MEL_BINS)
            melFrames.addLast(frame)
        }
        while (melFrames.size > MEL_FRAMES) melFrames.removeFirst()
        if (melFrames.size >= MEL_FRAMES) {
            var w = 0
            for (i in melFrames.size - MEL_FRAMES until melFrames.size) {
                melFrames[i].copyInto(embWindow, w, 0, MEL_BINS)
                w += MEL_BINS
            }
            try {
                // Copy the flat window into the rank-4 wrapper: [1,76,32,1].
                var v = 0
                for (f in 0 until MEL_FRAMES) {
                    val plane = embInArr[0][f]
                    for (b in 0 until MEL_BINS) plane[b][0] = embWindow[v++]
                }
                embedding.run(embInArr, embOutArr)
                System.arraycopy(embOutArr[0][0][0], 0, embOut, 0, EMBED_DIM)
                embeddings.addLast(embOut.copyOf())
            } catch (e: Throwable) {
                Log.e(TAG, "embedding model failed (input ${embWindow.size} floats, expected 76×32)", e)
            }
        }
        while (embeddings.size > EMB_MAX) embeddings.removeFirst()
        // Context for the next chunk = the last MEL_CONTEXT samples of this chunk.
        history = if (chunk.size >= MEL_CONTEXT) {
            chunk.copyOfRange(chunk.size - MEL_CONTEXT, chunk.size)
        } else {
            chunk.copyOf()
        }
    }

    private fun currentScore(): Float {
        if (embeddings.size < WAKE_FRAMES) return 0f
        fillWakeInput(embeddings, wakeIn, wakeInputIsTransposed)
        // Copy the flat buffer into the rank-3 wrapper matching the declared input:
        // [1,16,96] (frames×dim) or [1,96,16] (dim×frames, transposed exports).
        return if (wakeInputIsTransposed) {
            val dst = wakeInArrT[0]
            var v = 0
            for (d in 0 until EMBED_DIM) {
                val row = dst[d]
                for (k in 0 until WAKE_FRAMES) row[k] = wakeIn[v++]
            }
            wakeWord.run(wakeInArrT, wakeOutArr)
            wakeOutArr[0][0]
        } else {
            val dst = wakeInArr[0]
            var v = 0
            for (k in 0 until WAKE_FRAMES) {
                val row = dst[k]
                for (d in 0 until EMBED_DIM) row[d] = wakeIn[v++]
            }
            wakeWord.run(wakeInArr, wakeOutArr)
            wakeOutArr[0][0]
        }
    }

    private fun takeConfirmation(): Boolean =
        if (++confirmHits >= confirmationsNeeded) {
            confirmHits = 0
            true
        } else {
            false
        }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { embedding.close() }
        runCatching { wakeWord.close() }
    }

    companion object {
        private const val TAG = "OpenWakeWordDetector"
        const val WAKE_FRAMES = OWW_WAKE_FRAMES
        const val EMBED_DIM = OWW_EMBED_DIM
        private const val CHUNK_SAMPLES = 1280                  // 80 ms @ 16 kHz
        private const val MEL_CONTEXT = 480                     // 30 ms left context (160 * 3)
        private const val MEL_INPUT_SAMPLES = CHUNK_SAMPLES + MEL_CONTEXT // 1760
        private const val MEL_BINS = 32
        private const val MEL_FRAMES = 76                       // embedding window (frames)
        private const val MEL_FRAMES_PER_CHUNK = 8              // ~80 ms of mel frames / chunk
        /** Frames produced by the fixed 1760-sample melspectrogram window: (1760-512)/160 + 1. */
        private const val MAX_MEL_FRAMES = (MEL_INPUT_SAMPLES - 512) / 160 + 1 // = 8
        private const val EMB_MAX = 120                         // rolling embedding buffer (~10 s)

        /**
         * Maps the app `sensitivity` (0.2..0.9, higher = stricter) to a score threshold.
         * Monotonic increasing: higher sensitivity -> higher threshold -> stricter,
         * matching the UI description ("higher values are stricter"). Default
         * sensitivity 0.3 -> threshold 0.6 (unchanged historical default);
         * max sensitivity 0.9 -> threshold 0.95; min 0.2 -> 0.5.
         * (openWakeWord's own default is 0.5; we expose it through the existing UI knob.)
         */
        fun thresholdFor(sensitivity: Float): Float =
            (0.6f + (sensitivity - 0.3f)).coerceIn(0.5f, 0.95f)
    }
}
