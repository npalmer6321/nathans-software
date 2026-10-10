package selfforge.codegen

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

/**
 * A character-level recurrent neural network, written from scratch.
 * It learns to predict the next character of program text — i.e. it learns
 * the *shape* of code. Every [train] call rewrites its weights via
 * backpropagation through time.
 *
 * Architecture: one-hot char -> tanh recurrent hidden layer -> softmax vocab.
 */
class CharRNN(
    val vocab: List<Char>,
    val hiddenSize: Int = 96,
    var learningRate: Double = 0.05,
    seed: Long = 42L
) {
    val vocabSize: Int = vocab.size
    private val charToIdx: Map<Char, Int> = vocab.withIndex().associate { it.value to it.index }
    private val rng = Random(seed)

    // Weights (owned by this model, mutated by training).
    private val wIh: Array<DoubleArray> = Array(hiddenSize) { DoubleArray(vocabSize) { rng.nextDouble(-1.0, 1.0) * 0.08 } }
    private val wHh: Array<DoubleArray> = Array(hiddenSize) { DoubleArray(hiddenSize) { rng.nextDouble(-1.0, 1.0) * 0.08 } }
    private val bH = DoubleArray(hiddenSize)
    private val wHo: Array<DoubleArray> = Array(vocabSize) { DoubleArray(hiddenSize) { rng.nextDouble(-1.0, 1.0) * 0.08 } }
    private val bO = DoubleArray(vocabSize)

    fun weightCount(): Int = hiddenSize * vocabSize + hiddenSize * hiddenSize + hiddenSize +
            vocabSize * hiddenSize + vocabSize

    private fun softmax(z: DoubleArray): DoubleArray {
        val m = z.max()
        val e = DoubleArray(z.size) { exp(z[it] - m) }
        val s = e.sum()
        return DoubleArray(z.size) { e[it] / s }
    }

    private fun idxOf(c: Char): Int = charToIdx[c]
        ?: throw IllegalArgumentException("char '$c' not in vocab")

    /**
     * Train on a list of program texts (next-character prediction).
     * One SGD step per program, programs shuffled each epoch (seeded).
     */
    fun train(texts: List<String>, epochs: Int) {
        if (texts.isEmpty() || epochs <= 0) return
        val order = texts.indices.toMutableList()
        repeat(epochs) {
            order.shuffle(rng)
            for (oi in order) trainOne(texts[oi])
        }
    }

    private fun trainOne(text: String) {
        val seq = IntArray(text.length) { idxOf(text[it]) }
        val t = seq.size - 1
        if (t <= 0) return

        // Forward: h[-1] = 0
        val h = Array(t + 1) { DoubleArray(hiddenSize) } // h[s] = hidden after char s; h[t] unused
        val probs = Array(t) { DoubleArray(vocabSize) }
        var hPrev = DoubleArray(hiddenSize)
        for (s in 0 until t) {
            val a = DoubleArray(hiddenSize) { j ->
                var sum = bH[j] + wIh[j][seq[s]]
                for (k in 0 until hiddenSize) sum += wHh[j][k] * hPrev[k]
                sum
            }
            val hs = DoubleArray(hiddenSize) { j -> tanh(a[j]) }
            h[s] = hs
            val z = DoubleArray(vocabSize) { v ->
                var sum = bO[v]
                for (j in 0 until hiddenSize) sum += wHo[v][j] * hs[j]
                sum
            }
            probs[s] = softmax(z)
            hPrev = hs
        }

        // Backward through time.
        val dWIh = Array(hiddenSize) { DoubleArray(vocabSize) }
        val dWHh = Array(hiddenSize) { DoubleArray(hiddenSize) }
        val dBH = DoubleArray(hiddenSize)
        val dWHo = Array(vocabSize) { DoubleArray(hiddenSize) }
        val dBO = DoubleArray(vocabSize)
        var dhNext = DoubleArray(hiddenSize)

        for (s in t - 1 downTo 0) {
            val dz = probs[s].copyOf()
            dz[seq[s + 1]] -= 1.0 // dL/dz for softmax + cross-entropy
            for (v in 0 until vocabSize) {
                dBO[v] += dz[v]
                for (j in 0 until hiddenSize) dWHo[v][j] += dz[v] * h[s][j]
            }
            val hPrevS = if (s == 0) DoubleArray(hiddenSize) else h[s - 1]
            val dh = DoubleArray(hiddenSize) { j ->
                var sum = dhNext[j]
                for (v in 0 until vocabSize) sum += wHo[v][j] * dz[v]
                val th = h[s][j]
                sum * (1.0 - th * th)
            }
            for (j in 0 until hiddenSize) {
                dBH[j] += dh[j]
                dWIh[j][seq[s]] += dh[j]
                for (k in 0 until hiddenSize) dWHh[j][k] += dh[j] * hPrevS[k]
            }
            val newDhNext = DoubleArray(hiddenSize) { k ->
                var sum = 0.0
                for (j in 0 until hiddenSize) sum += wHh[j][k] * dh[j]
                sum
            }
            dhNext = newDhNext
        }

        // Gradient clipping + SGD update (gradients averaged over sequence steps).
        val invT = 1.0 / t
        fun clip(v: Double) = v.coerceIn(-5.0, 5.0)
        for (j in 0 until hiddenSize) {
            bH[j] -= learningRate * clip(dBH[j] * invT)
            for (v in 0 until vocabSize) {
                wIh[j][v] -= learningRate * clip(dWIh[j][v] * invT)
                wHo[v][j] -= learningRate * clip(dWHo[v][j] * invT)
            }
            for (k in 0 until hiddenSize) wHh[j][k] -= learningRate * clip(dWHh[j][k] * invT)
        }
        for (v in 0 until vocabSize) bO[v] -= learningRate * clip(dBO[v] * invT)
    }

    /** Average cross-entropy loss per character (lower = better model of the text). */
    fun loss(texts: List<String>): Double {
        var total = 0.0
        var n = 0
        for (text in texts) {
            val seq = IntArray(text.length) { idxOf(text[it]) }
            var hPrev = DoubleArray(hiddenSize)
            for (s in 0 until seq.size - 1) {
                val hs = DoubleArray(hiddenSize) { j ->
                    var sum = bH[j] + wIh[j][seq[s]]
                    for (k in 0 until hiddenSize) sum += wHh[j][k] * hPrev[k]
                    tanh(sum)
                }
                val z = DoubleArray(vocabSize) { v ->
                    var sum = bO[v]
                    for (j in 0 until hiddenSize) sum += wHo[v][j] * hs[j]
                    sum
                }
                val p = softmax(z)
                total += -ln(p[seq[s + 1]].coerceAtLeast(1e-12))
                n++
                hPrev = hs
            }
        }
        return if (n == 0) 0.0 else total / n
    }

    /**
     * Generate program text: feed [seed], then sample [length] more characters.
     * Higher temperature = more creative/riskier.
     */
    fun sample(seed: String, length: Int, temperature: Double, rng: Random): String {
        require(seed.isNotEmpty())
        val sb = StringBuilder(seed)
        // Warm up: hidden state after the whole seed.
        var hPrev = DoubleArray(hiddenSize)
        for (ch in seed) {
            hPrev = step(idxOf(ch), hPrev)
        }
        repeat(length) {
            val p = probsFromState(hPrev, temperature)
            // Sample from the distribution.
            var r = rng.nextDouble()
            var v = 0
            while (v < vocabSize - 1 && r > p[v]) {
                r -= p[v]
                v++
            }
            sb.append(vocab[v])
            hPrev = step(v, hPrev)
            if (vocab[v] == '\n' && sb.count { it == '\n' } >= 8) return sb.toString()
        }
        return sb.toString()
    }

    private fun step(inputIdx: Int, hPrev: DoubleArray): DoubleArray =
        DoubleArray(hiddenSize) { j ->
            var sum = bH[j] + wIh[j][inputIdx]
            for (k in 0 until hiddenSize) sum += wHh[j][k] * hPrev[k]
            tanh(sum)
        }

    /** Next-char distribution given the current hidden state (no double-stepping). */
    private fun probsFromState(hPrev: DoubleArray, temperature: Double): DoubleArray {
        val t = temperature.coerceAtLeast(0.05)
        val z = DoubleArray(vocabSize) { v ->
            var sum = bO[v]
            for (j in 0 until hiddenSize) sum += wHo[v][j] * hPrev[j]
            sum / t
        }
        return softmax(z)
    }

    /** Serialize weights (for saving the code model). */
    fun exportWeights(): String {
        val sb = StringBuilder()
        sb.append("$vocabSize,$hiddenSize\n")
        fun dump(m: Array<DoubleArray>) { for (row in m) sb.append(row.joinToString(",")).append('\n') }
        dump(wIh); dump(wHh); dump(wHo)
        sb.append(bH.joinToString(",")).append('\n')
        sb.append(bO.joinToString(",")).append('\n')
        return sb.toString()
    }

    companion object {
        fun importWeights(vocab: List<Char>, data: String, learningRate: Double = 0.05): CharRNN {
            val lines = data.trim().lines()
            val (vs, hs) = lines[0].split(",").map { it.toInt() }
            require(vs == vocab.size)
            val net = CharRNN(vocab, hs, learningRate, seed = 0L)
            var line = 1
            fun load(m: Array<DoubleArray>) {
                for (r in m.indices) {
                    val vals = lines[line++].split(",").map { it.toDouble() }
                    for (c in vals.indices) m[r][c] = vals[c]
                }
            }
            load(net.wIh); load(net.wHh); load(net.wHo)
            lines[line++].split(",").map { it.toDouble() }.toDoubleArray().copyInto(net.bH)
            lines[line++].split(",").map { it.toDouble() }.toDoubleArray().copyInto(net.bO)
            return net
        }
    }
}
