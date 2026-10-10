package selfforge

import kotlin.math.exp
import kotlin.math.tanh
import kotlin.random.Random

/** One labeled training example. */
data class Example(val input: DoubleArray, val label: Int)

/**
 * A small fully-connected neural network implemented from scratch —
 * no ML frameworks. It owns its weights and changes them every time
 * [trainBatch] runs (on-device learning, no server involved).
 *
 * Architecture: input -> tanh hidden layers -> softmax output.
 * Loss: cross-entropy. Optimizer: full-batch gradient descent.
 */
class NeuralNet(
    val layerSizes: IntArray,
    var learningRate: Double = 0.05,
    seed: Long = 42L
) {
    private val rng = Random(seed)

    /**
     * weights[layer][from][to], layer in 1..last.
     * `from` index 0 is the bias neuron (always 1.0).
     */
    private val weights: Array<Array<DoubleArray>> = Array(layerSizes.size) { l ->
        if (l == 0) emptyArray()
        else Array(layerSizes[l - 1] + 1) { i ->
            // Xavier-style init scaled for tanh; bias row (i==0) starts smaller.
            val scale = if (i == 0) 0.1 else kotlin.math.sqrt(1.0 / layerSizes[l - 1])
            DoubleArray(layerSizes[l]) { rng.nextDouble(-1.0, 1.0) * scale }
        }
    }

    private data class ForwardCache(val activations: List<DoubleArray>, val preActivations: List<DoubleArray>)

    private fun softmax(z: DoubleArray): DoubleArray {
        val m = z.max()
        val e = DoubleArray(z.size) { exp(z[it] - m) }
        val s = e.sum()
        return DoubleArray(z.size) { e[it] / s }
    }

    private fun forward(input: DoubleArray): ForwardCache {
        require(input.size == layerSizes[0]) { "expected ${layerSizes[0]} inputs, got ${input.size}" }
        val a = mutableListOf(input)
        val z = mutableListOf<DoubleArray>(DoubleArray(0))
        for (l in 1 until layerSizes.size) {
            val prev = a[l - 1]
            val zl = DoubleArray(layerSizes[l]) { j ->
                var sum = weights[l][0][j] // bias
                for (i in prev.indices) sum += weights[l][i + 1][j] * prev[i]
                sum
            }
            z.add(zl)
            a.add(if (l == layerSizes.size - 1) softmax(zl) else DoubleArray(zl.size) { j -> tanh(zl[j]) })
        }
        return ForwardCache(a, z)
    }

    /** Probability distribution over output classes. */
    fun predictProbs(input: DoubleArray): DoubleArray = forward(input).activations.last()

    /** Most likely class index. */
    fun predict(input: DoubleArray): Int {
        val p = predictProbs(input)
        var best = 0
        for (j in 1 until p.size) if (p[j] > p[best]) best = j
        return best
    }

    /** Fraction of examples classified correctly. */
    fun accuracy(examples: List<Example>): Double {
        if (examples.isEmpty()) return 0.0
        return examples.count { predict(it.input) == it.label }.toDouble() / examples.size
    }

    /**
     * Train the model: mini-batch SGD with backprop. Every call literally
     * rewrites the model's weights — this is the "changes its own weights" part.
     * Shuffling each epoch (seeded RNG, so runs are reproducible) keeps the
     * optimizer from getting stuck in degenerate solutions.
     */
    fun trainBatch(examples: List<Example>, epochs: Int, batchSize: Int = 16) {
        if (examples.isEmpty() || epochs <= 0) return
        val last = layerSizes.size - 1
        val order = examples.indices.toMutableList()
        repeat(epochs) {
            order.shuffle(rng)
            var start = 0
            while (start < order.size) {
                val end = minOf(start + batchSize, order.size)
                val grads = Array(layerSizes.size) { l ->
                    if (l == 0) emptyArray()
                    else Array(layerSizes[l - 1] + 1) { DoubleArray(layerSizes[l]) }
                }
                for (oi in start until end) {
                    val ex = examples[order[oi]]
                    val (a, z) = forward(ex.input)
                    val deltas = arrayOfNulls<DoubleArray>(layerSizes.size)
                    // Output layer: softmax + cross-entropy => delta = p - oneHot
                    val out = a[last]
                    deltas[last] = DoubleArray(out.size) { j -> out[j] - if (j == ex.label) 1.0 else 0.0 }
                    // Hidden layers, back to layer 1
                    for (l in last - 1 downTo 1) {
                        val zl = z[l]
                        deltas[l] = DoubleArray(layerSizes[l]) { j ->
                            var s = 0.0
                            val next = deltas[l + 1]!!
                            for (k in next.indices) s += weights[l + 1][j + 1][k] * next[k]
                            val t = tanh(zl[j])
                            s * (1.0 - t * t)
                        }
                    }
                    // Accumulate gradients
                    for (l in 1..last) {
                        val prev = a[l - 1]
                        val dl = deltas[l]!!
                        for (i in 0..layerSizes[l - 1]) {
                            val ai = if (i == 0) 1.0 else prev[i - 1]
                            for (j in dl.indices) grads[l][i][j] += ai * dl[j]
                        }
                    }
                }
                val n = (end - start).toDouble()
                for (l in 1..last) {
                    for (i in grads[l].indices) {
                        for (j in grads[l][i].indices) {
                            weights[l][i][j] -= learningRate * grads[l][i][j] / n
                        }
                    }
                }
                start = end
            }
        }
    }

    /** Total number of trainable weights (proof the model is real, not a stub). */
    fun weightCount(): Int {
        var c = 0
        for (l in 1 until layerSizes.size) c += (layerSizes[l - 1] + 1) * layerSizes[l]
        return c
    }

    /** Overwrite this net's weights with another net's (same architecture). Used to restore the best snapshot. */
    fun copyWeightsFrom(other: NeuralNet) {
        require(other.layerSizes.contentEquals(layerSizes)) { "architecture mismatch" }
        for (l in 1 until layerSizes.size) {
            for (i in weights[l].indices) {
                other.weights[l][i].copyInto(weights[l][i])
            }
        }
    }

    /** Serialize architecture + weights to a string (for saving to disk). */
    fun exportWeights(): String {
        val sb = StringBuilder()
        sb.append(layerSizes.joinToString(",")).append('\n')
        for (l in 1 until layerSizes.size) {
            for (i in weights[l].indices) {
                sb.append(weights[l][i].joinToString(",")).append('\n')
            }
        }
        return sb.toString()
    }

    companion object {
        /** Rebuild a net from [exportWeights] output (same seed not needed). */
        fun importWeights(data: String, learningRate: Double = 0.05): NeuralNet {
            val lines = data.trim().lines()
            val sizes = lines[0].split(",").map { it.toInt() }.toIntArray()
            val net = NeuralNet(sizes, learningRate, seed = 0L)
            var line = 1
            for (l in 1 until sizes.size) {
                for (i in 0..sizes[l - 1]) {
                    val vals = lines[line++].split(",").map { it.toDouble() }
                    for (j in vals.indices) net.weights[l][i][j] = vals[j]
                }
            }
            return net
        }
    }
}
