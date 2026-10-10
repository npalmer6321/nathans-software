package forgegpt

import java.io.File
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 * Engine — a small, UI-agnostic facade over the ForgeGPT core (GPT,
 * Tokenizer, Generate, GradCheck, Trainer).
 *
 * The CLI (Main/Train/Generate) and the Swing GUI both sit on top of the
 * same machinery; the GUI uses this object so it never has to scrape
 * console output or duplicate sampling/evaluation logic.
 *
 * Everything here is thread-safe for one-model/one-worker use as long as
 * each caller gets its own [GPT.Worker] via [LoadedModel.newWorker].
 */

/** A model + tokenizer pair loaded from disk, ready for work. */
class LoadedModel(
    val model: GPT,
    val tokenizer: Tokenizer,
    val modelFile: File,
    val vocabFile: File
) {
    val config: GPTConfig get() = model.config
    val numParams: Long get() = model.numParams

    fun newWorker(): GPT.Worker = model.newWorker()

    /** Human-readable one-paragraph summary used in several GUI tabs. */
    fun describe(): String {
        val c = config
        return "vocab=${c.vocab}, dModel=${c.dModel}, layers=${c.nLayers}, heads=${c.nHeads}, " +
            "headDim=${c.headDim}, seqLen=${c.seqLen}, ffn=${c.ffnHidden}, " +
            "params=${"%,d".format(numParams)}"
    }
}

/** Breakdown of parameter counts by component, for the Model Lab tab. */
data class ParamBreakdown(
    val totalParams: Long,
    val tokenEmbedding: Long,
    val positionEmbedding: Long,
    val attentionPerLayer: Long,
    val mlpPerLayer: Long,
    val layernormsTotal: Long,
    val outputHead: Long,
    val numLayers: Int
) {
    /** Rows of (label, count) for display in a table. */
    fun rows(): List<Pair<String, Long>> {
        val rows = ArrayList<Pair<String, Long>>()
        rows += "Token embedding (vocab x dModel)" to tokenEmbedding
        rows += "Position embedding (seqLen x dModel)" to positionEmbedding
        rows += "Attention per layer (QKV + out)" to attentionPerLayer
        rows += "MLP per layer (up + down)" to mlpPerLayer
        rows += "All layers' attention (x$numLayers)" to attentionPerLayer * numLayers
        rows += "All layers' MLP (x$numLayers)" to mlpPerLayer * numLayers
        rows += "LayerNorm gains/biases (all)" to layernormsTotal
        rows += "Output head (dModel x vocab + bias)" to outputHead
        rows += "TOTAL" to totalParams
        return rows
    }
}

/** Statistics about a tokenized corpus, for the Corpus tab. */
data class CorpusStats(
    val sourceName: String,
    val bytes: Long,
    val chars: Int,
    val wordsApprox: Int,
    val tokens: Int,
    val vocabSize: Int,
    val uniqueTokensInText: Int,
    val typeTokenRatio: Double,
    val unkRate: Double,
    val avgTokensPerLine: Double,
    val lines: Int,
    val topTokens: List<Pair<String, Int>>
)

/** Result of a loss/perplexity evaluation pass. */
data class EvalResult(
    val windows: Int,
    val meanLoss: Float,
    val perplexity: Double,
    val tokensScored: Int,
    val bestWindowLoss: Float,
    val worstWindowLoss: Float,
    val elapsedMs: Long
)

/** One generated sample with the settings that produced it. */
data class GenerationResult(
    val prompt: String,
    val fullText: String,
    val generatedTokens: Int,
    val temperature: Float,
    val topK: Int,
    val elapsedMs: Long,
    val stoppedOnEos: Boolean
) {
    val tokensPerSec: Double
        get() = if (elapsedMs > 0) generatedTokens * 1000.0 / elapsedMs else 0.0
}

object Engine {

    /** Load a model + vocab from disk. Throws IllegalArgumentException on bad files. */
    fun loadModel(modelFile: File, vocabFile: File): LoadedModel {
        require(modelFile.isFile) { "model file not found: ${modelFile.path}" }
        require(vocabFile.isFile) { "vocab file not found: ${vocabFile.path}" }
        val model = GPT.load(modelFile)
        val tok = Tokenizer.load(vocabFile)
        require(tok.vocabSize == model.config.vocab) {
            "vocab size ${tok.vocabSize} does not match model vocab ${model.config.vocab}"
        }
        return LoadedModel(model, tok, modelFile, vocabFile)
    }

    /**
     * Compute a parameter-count breakdown from the config + live params.
     * Counts are derived from the actual [Param] tensors so they always
     * match the model in memory.
     */
    fun paramBreakdown(model: GPT): ParamBreakdown {
        val c = model.config
        fun sizeOf(name: String): Long =
            model.params.firstOrNull { it.name == name }?.data?.size?.toLong() ?: 0L
        val tokEmb = sizeOf("tokEmb")
        val posEmb = sizeOf("posEmb")
        var lnTotal = sizeOf("lnFg") + sizeOf("lnFb")
        var attnPerLayer = 0L
        var mlpPerLayer = 0L
        for (l in 0 until c.nLayers) {
            lnTotal += sizeOf("ln1g.$l") + sizeOf("ln1b.$l") + sizeOf("ln2g.$l") + sizeOf("ln2b.$l")
            if (l == 0) {
                attnPerLayer = sizeOf("qkvW.$l") + sizeOf("qkvB.$l") + sizeOf("outW.$l") + sizeOf("outB.$l")
                mlpPerLayer = sizeOf("ffn1W.$l") + sizeOf("ffn1B.$l") + sizeOf("ffn2W.$l") + sizeOf("ffn2B.$l")
            }
        }
        val head = sizeOf("headW") + sizeOf("headB")
        return ParamBreakdown(
            totalParams = model.numParams,
            tokenEmbedding = tokEmb,
            positionEmbedding = posEmb,
            attentionPerLayer = attnPerLayer,
            mlpPerLayer = mlpPerLayer,
            layernormsTotal = lnTotal,
            outputHead = head,
            numLayers = c.nLayers
        )
    }

    /**
     * Streaming generation: calls [onText] with the decoded text so far
     * after every token, giving the UI a typewriter effect. Returns the
     * final [GenerationResult]. Generation runs on the caller's thread;
     * [shouldStop] is polled between tokens for a responsive Stop button.
     */
    fun generateStreaming(
        loaded: LoadedModel,
        worker: GPT.Worker,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        seed: Long,
        shouldStop: () -> Boolean = { false },
        onText: (String) -> Unit = {}
    ): GenerationResult {
        val tok = loaded.tokenizer
        val model = loaded.model
        val rng = Random(seed)
        val t0 = System.currentTimeMillis()

        val ids = ArrayList<Int>()
        val enc = tok.encode(prompt.ifBlank { "the" })
        for (x in enc) ids.add(x)
        if (ids.isEmpty()) ids.add(Tokenizer.BOS)
        val promptTokenCount = ids.size

        var stoppedOnEos = false
        var generated = 0
        while (generated < maxTokens) {
            if (shouldStop()) break
            val context = ids.takeLast(model.config.seqLen).toIntArray()
            val logits = worker.forwardLogitsLast(context)
            val next = sampleToken(logits, temperature, topK, rng)
            ids.add(next)
            generated++
            if (next == Tokenizer.EOS) {
                stoppedOnEos = true
                break
            }
            onText(tok.decode(ids.drop(promptTokenCount)))
        }
        val full = tok.decode(ids)
        onText(tok.decode(ids.drop(promptTokenCount)))
        return GenerationResult(
            prompt = prompt,
            fullText = full,
            generatedTokens = generated,
            temperature = temperature,
            topK = topK,
            elapsedMs = System.currentTimeMillis() - t0,
            stoppedOnEos = stoppedOnEos
        )
    }

    /** Temperature + top-k sampling, shared with [Generate]'s private copy. */
    fun sampleToken(logits: FloatArray, temperature: Float, topK: Int, rng: Random): Int {
        if (temperature <= 0f) {
            var best = 0
            for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
            return best
        }
        val inv = 1f / temperature
        val k = topK.coerceIn(1, logits.size)
        val order = logits.indices.sortedByDescending { logits[it] }.take(k)
        var m = Float.NEGATIVE_INFINITY
        for (i in order) {
            val v = logits[i] * inv
            if (v > m) m = v
        }
        var s = 0.0
        val probs = DoubleArray(order.size)
        for ((idx, i) in order.withIndex()) {
            val e = exp((logits[i] * inv - m).toDouble())
            probs[idx] = e
            s += e
        }
        var r = rng.nextDouble() * s
        for (idx in order.indices) {
            r -= probs[idx]
            if (r <= 0) return order[idx]
        }
        return order.last()
    }

    /**
     * Score text with the model: slide fixed windows over the token stream
     * and average next-token cross-entropy. Lower loss / perplexity = the
     * model finds the text more predictable.
     */
    fun evaluate(
        loaded: LoadedModel,
        worker: GPT.Worker,
        text: String,
        maxWindows: Int = 24
    ): EvalResult {
        val tok = loaded.tokenizer
        val t0 = System.currentTimeMillis()
        val ids = tok.encodeDocument(text)
        val seqLen = loaded.config.seqLen
        require(ids.size > seqLen) { "text too short to evaluate (${ids.size} tokens, need > $seqLen)" }

        val windowsAvail = ids.size - seqLen
        val windows = minOf(maxWindows, windowsAvail)
        // Spread windows evenly so a long pasted text is sampled, not just its start.
        val stride = if (windows > 1) (windowsAvail - 1).toDouble() / (windows - 1) else 1.0

        var lossSum = 0.0
        var best = Float.MAX_VALUE
        var worst = Float.MIN_VALUE
        val seq = IntArray(seqLen + 1)
        for (w in 0 until windows) {
            val start = (w * stride).toInt().coerceAtMost(windowsAvail - 1)
            ids.copyInto(seq, 0, start, start + seqLen + 1)
            val loss = worker.forward(seq)
            lossSum += loss.toDouble()
            if (loss < best) best = loss
            if (loss > worst) worst = loss
        }
        val mean = (lossSum / windows).toFloat()
        return EvalResult(
            windows = windows,
            meanLoss = mean,
            perplexity = exp(mean.toDouble()),
            tokensScored = windows * seqLen,
            bestWindowLoss = best,
            worstWindowLoss = worst,
            elapsedMs = System.currentTimeMillis() - t0
        )
    }

    /** Corpus statistics for a raw text blob scored against a tokenizer. */
    fun corpusStats(sourceName: String, text: String, tokenizer: Tokenizer): CorpusStats {
        val toks = Tokenizer.tokenizeText(text)
        val counts = HashMap<String, Int>()
        for (t in toks) counts[t] = (counts[t] ?: 0) + 1
        var unk = 0
        for (t in toks) if (!tokenizer.tokenToId.containsKey(t)) unk++
        val lines = if (text.isEmpty()) 0 else text.count { it == '\n' } + 1
        val top = counts.entries.sortedByDescending { it.value }.take(25).map { it.key to it.value }
        return CorpusStats(
            sourceName = sourceName,
            bytes = text.toByteArray(Charsets.UTF_8).size.toLong(),
            chars = text.length,
            wordsApprox = toks.size,
            tokens = toks.size,
            vocabSize = tokenizer.vocabSize,
            uniqueTokensInText = counts.size,
            typeTokenRatio = if (toks.isEmpty()) 0.0 else counts.size.toDouble() / toks.size,
            unkRate = if (toks.isEmpty()) 0.0 else unk.toDouble() / toks.size,
            avgTokensPerLine = if (lines == 0) 0.0 else toks.size.toDouble() / lines,
            lines = lines,
            topTokens = top
        )
    }

    /** Estimated on-disk size of a model file from its parameter count (float32 + header). */
    fun estimatedModelBytes(model: GPT): Long = model.numParams * 4L + 4096L

    /** Random-guessing loss for a vocab, for "how much has it learned" displays. */
    fun randomLoss(vocabSize: Int): Double = ln(vocabSize.toDouble())
}
