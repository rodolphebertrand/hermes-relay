package com.hermesandroid.relay.wake

import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.TensorFlowLiteModel

/**
 * openWakeWord TFLite wake-word detector ([WakeWordDetector] implementation).
 *
 * Pipeline (faithful to `openwakeword.utils.AudioPreprocessor`):
 *   1. Audio is buffered into 1280-sample (80 ms @ 16 kHz) chunks.
 *   2. Each chunk is run through the melspectrogram model with 480 samples of left
 *      context; the new mel frames are appended to a rolling buffer (kept at <= 76 frames).
 *   3. The post-mel transform `spec = spec / 10 + 2` is applied (MANDATORY — matches
 *      openwakeword/utils.py) before the embedding model.
 *   4. The last 76 mel frames form the window for the embedding model -> one 96-dim
 *      embedding, appended to a rolling buffer (maxlen 120).
 *   5. The wake-word classifier consumes the LAST 16 embeddings ([1,16,96]) -> score 0..1.
 *   6. A detection fires when `score >= threshold` for `confirmationFrames` consecutive
 *      80 ms windows (debounce against single-frame spikes).
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
) : WakeWordDetector {

    private val options = Interpreter.Options().apply {
        numThreads = 2
        // CPU inference: reliable and fast enough for the 80 ms cadence. A LiteRT GPU
        // delegate can be layered on later as a measured optimization.
    }

    private val melspec =
        Interpreter(TensorFlowLiteModel.load(files.melspectrogram.absolutePath), options)
    private val embedding =
        Interpreter(TensorFlowLiteModel.load(files.embedding.absolutePath), options)
    private val wakeWord =
        Interpreter(TensorFlowLiteModel.load(files.wakeWord.absolutePath), options)

    init {
        // The melspectrogram model takes a variable-length input; pin it to our fixed
        // 1760-sample window (480 context + 1280 chunk) so we can feed it without
        // re-resizing on every chunk.
        melspec.resizeInputTensor(0, intArrayOf(1, MEL_INPUT_SAMPLES), true)
        melspec.allocateTensors()
        embedding.allocateTensors()
        wakeWord.allocateTensors()
    }

    // --- Streaming buffers (mirror openwakeword.utils.AudioPreprocessor) ---
    private val melIn = FloatArray(MEL_INPUT_SAMPLES)
    private var history = ShortArray(0)              // last MEL_CONTEXT samples processed
    private val melFrames = ArrayDeque<FloatArray>() // rolling 76 frames of 32-dim
    private val embWindow = FloatArray(MEL_FRAMES * MEL_BINS)
    private val embOut = FloatArray(EMBED_DIM)
    private val embeddings = ArrayDeque<FloatArray>() // rolling 120 embeddings of 96-dim
    private val wakeIn = FloatArray(WAKE_FRAMES * EMBED_DIM)
    private var remainder = ShortArray(0)            // < 1280 leftover samples
    private var confirmHits = 0
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
        return currentScore() >= threshold && takeConfirmation()
    }

    private fun processChunk(chunk: ShortArray) {
        // Build the melspectrogram input: [480 left context (zero-padded)] + [1280 chunk].
        for (i in 0 until MEL_CONTEXT) melIn[i] = 0f
        val ctx = history.size.coerceAtMost(MEL_CONTEXT)
        for (i in 0 until ctx) melIn[MEL_CONTEXT - ctx + i] = history[history.size - ctx + i].toFloat()
        for (i in chunk.indices) melIn[MEL_CONTEXT + i] = chunk[i].toFloat()
        melspec.copyInputFromArray(0, melIn)
        melspec.invoke()
        val spec = melspec.getOutputArray(0) as FloatArray
        val frames = spec.size / MEL_BINS
        val newFrames = minOf(frames, MEL_FRAMES_PER_CHUNK)
        for (f in (frames - newFrames) until frames) {
            val frame = FloatArray(MEL_BINS)
            spec.copyInto(frame, 0, f * MEL_BINS, (f + 1) * MEL_BINS)
            for (b in frame.indices) frame[b] = frame[b] / 10f + 2f // mandatory post-mel
            melFrames.addLast(frame)
        }
        while (melFrames.size > MEL_FRAMES) melFrames.removeFirst()
        if (melFrames.size >= MEL_FRAMES) {
            var w = 0
            for (i in melFrames.size - MEL_FRAMES until melFrames.size) {
                melFrames[i].copyInto(embWindow, w, 0, MEL_BINS)
                w += MEL_BINS
            }
            embedding.copyInputFromArray(0, embWindow)
            embedding.invoke()
            (embedding.getOutputArray(0) as FloatArray).copyInto(embOut)
            embeddings.addLast(embOut.copyOf())
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
        var w = 0
        for (i in embeddings.size - WAKE_FRAMES until embeddings.size) {
            embeddings[i].copyInto(wakeIn, w, 0, EMBED_DIM)
            w += EMBED_DIM
        }
        wakeWord.copyInputFromArray(0, wakeIn)
        wakeWord.invoke()
        return (wakeWord.getOutputArray(0) as FloatArray).firstOrNull() ?: 0f
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
        runCatching { melspec.close() }
        runCatching { embedding.close() }
        runCatching { wakeWord.close() }
    }

    companion object {
        private const val TAG = "OpenWakeWordDetector"
        private const val CHUNK_SAMPLES = 1280                  // 80 ms @ 16 kHz
        private const val MEL_CONTEXT = 480                     // 30 ms left context (160 * 3)
        private const val MEL_INPUT_SAMPLES = CHUNK_SAMPLES + MEL_CONTEXT // 1760
        private const val MEL_BINS = 32
        private const val MEL_FRAMES = 76                       // embedding window (frames)
        private const val MEL_FRAMES_PER_CHUNK = 8              // ~80 ms of mel frames / chunk
        private const val EMBED_DIM = 96
        private const val WAKE_FRAMES = 16                      // wake-word window (embeddings)
        private const val EMB_MAX = 120                         // rolling embedding buffer (~10 s)

        /**
         * Maps the app `sensitivity` (0.2..0.9, higher = stricter) to a score threshold.
         * Default sensitivity 0.3 -> threshold 0.6; max sensitivity 0.9 -> 0.3.
         * (openWakeWord's own default is 0.5; we expose it through the existing UI knob.)
         */
        fun thresholdFor(sensitivity: Float): Float =
            (0.9f - sensitivity).coerceIn(0.1f, 0.9f)
    }
}
