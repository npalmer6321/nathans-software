package forgegpt

import kotlin.math.exp
import kotlin.random.Random

/**
 * Causal text generation: feed the prompt through the model, then repeatedly
 * sample the next token from the last position's logits (temperature + top-k)
 * and append it to the context.
 */
object Generate {
    data class Args(
        val model: String = "model.fg",
        val vocab: String = "vocab.txt",
        val prompt: String = "once upon a time",
        val maxTokens: Int = 80,
        val temperature: Float = 0.8f,
        val topK: Int = 40,
        val seed: Long = 7L
    )

    fun parse(argv: List<String>): Args {
        var a = Args()
        val it = argv.iterator()
        while (it.hasNext()) {
            when (val k = it.next()) {
                "--model" -> a = a.copy(model = it.next())
                "--vocab" -> a = a.copy(vocab = it.next())
                "--prompt" -> a = a.copy(prompt = it.next())
                "--max-tokens" -> a = a.copy(maxTokens = it.next().toInt())
                "--temperature" -> a = a.copy(temperature = it.next().toFloat())
                "--top-k" -> a = a.copy(topK = it.next().toInt())
                "--seed" -> a = a.copy(seed = it.next().toLong())
                else -> println("generate: unknown arg $k (ignored)")
            }
        }
        return a
    }

    fun run(args: Args) {
        println("Loading model: ${args.model}")
        val model = GPT.load(java.io.File(args.model))
        val tok = Tokenizer.load(java.io.File(args.vocab))
        val worker = model.newWorker()
        val text = sample(
            model, tok, worker,
            prompt = args.prompt, maxTokens = args.maxTokens,
            temperature = args.temperature, topK = args.topK,
            rng = Random(args.seed)
        )
        println()
        println(text)
    }

    fun sample(
        model: GPT, tok: Tokenizer, worker: GPT.Worker,
        prompt: String, maxTokens: Int, temperature: Float, topK: Int, rng: Random
    ): String {
        val ids = ArrayList<Int>()
        val enc = tok.encode(prompt.ifBlank { "the" })
        for (x in enc) ids.add(x)
        if (ids.isEmpty()) ids.add(Tokenizer.BOS)

        repeat(maxTokens) {
            val context = ids.takeLast(model.config.seqLen).toIntArray()
            val logits = worker.forwardLogitsLast(context)
            val next = sampleToken(logits, temperature, topK, rng)
            ids.add(next)
            if (next == Tokenizer.EOS) return@repeat
        }
        return tok.decode(ids)
    }

    private fun sampleToken(logits: FloatArray, temperature: Float, topK: Int, rng: Random): Int {
        if (temperature <= 0f) {
            var best = 0
            for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
            return best
        }
        val inv = 1f / temperature
        // top-k pre-filter
        val order = logits.indices.sortedByDescending { logits[it] }.take(topK.coerceAtMost(logits.size))
        var m = Float.NEGATIVE_INFINITY
        for (i in order) {
            val v = logits[i] * inv
            if (v > m) m = v
        }
        var s = 0.0
        val probs = DoubleArray(order.size)
        for ((k, i) in order.withIndex()) {
            val e = exp((logits[i] * inv - m).toDouble())
            probs[k] = e
            s += e
        }
        var r = rng.nextDouble() * s
        for (k in order.indices) {
            r -= probs[k]
            if (r <= 0) return order[k]
        }
        return order.last()
    }
}
