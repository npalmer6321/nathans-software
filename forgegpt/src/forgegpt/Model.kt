package forgegpt

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.sqrt
import kotlin.random.Random

/** Model hyperparameters. */
data class GPTConfig(
    val vocab: Int,
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val seqLen: Int,
    val ffnHidden: Int,
    val seed: Long = 42L
) {
    val headDim: Int get() = dModel / nHeads
    init {
        require(dModel % nHeads == 0) { "dModel must be divisible by nHeads" }
    }
}

/** A named parameter tensor (flat row-major). */
class Param(val name: String, val data: FloatArray)

/**
 * Decoder-only transformer (GPT family), pre-norm blocks:
 *
 *   x = tokEmb + posEmb
 *   repeat L: x += Attn(LN(x)); x += MLP(LN(x))     (causal self-attention)
 *   logits = LN(x) @ headW + headB
 *
 * Pure Kotlin, no frameworks. Owns its parameters in a fixed order so
 * weights can be saved/loaded and optimized by [Adam].
 */
class GPT(val config: GPTConfig) {
    private val rng = Random(config.seed)
    val params = ArrayList<Param>()

    private fun param(name: String, size: Int, init: (Int) -> Float): FloatArray {
        val a = FloatArray(size, init)
        params.add(Param(name, a))
        return a
    }

    private fun randn(): Float {
        // Box-Muller
        var u = 0.0
        while (u == 0.0) u = rng.nextDouble()
        val v = rng.nextDouble()
        return (sqrt(-2.0 * kotlin.math.ln(u)) * kotlin.math.cos(2.0 * Math.PI * v)).toFloat()
    }

    private fun xavier(fanIn: Int, fanOut: Int): Float {
        val lim = sqrt(6.0 / (fanIn + fanOut)).toFloat()
        return rng.nextFloat() * 2f * lim - lim
    }

    // ---- parameters (fixed order = save/load order) ----
    val tokEmb: FloatArray = param("tokEmb", config.vocab * config.dModel) { randn() * 0.02f }
    val posEmb: FloatArray = param("posEmb", config.seqLen * config.dModel) { randn() * 0.02f }

    val ln1g: Array<FloatArray> = Array(config.nLayers) { l -> param("ln1g.$l", config.dModel) { 1f } }
    val ln1b: Array<FloatArray> = Array(config.nLayers) { l -> param("ln1b.$l", config.dModel) { 0f } }
    val qkvW: Array<FloatArray> = Array(config.nLayers) { l ->
        param("qkvW.$l", config.dModel * 3 * config.dModel) { xavier(config.dModel, 3 * config.dModel) }
    }
    val qkvB: Array<FloatArray> = Array(config.nLayers) { l -> param("qkvB.$l", 3 * config.dModel) { 0f } }
    val outW: Array<FloatArray> = Array(config.nLayers) { l ->
        param("outW.$l", config.dModel * config.dModel) { xavier(config.dModel, config.dModel) }
    }
    val outB: Array<FloatArray> = Array(config.nLayers) { l -> param("outB.$l", config.dModel) { 0f } }
    val ln2g: Array<FloatArray> = Array(config.nLayers) { l -> param("ln2g.$l", config.dModel) { 1f } }
    val ln2b: Array<FloatArray> = Array(config.nLayers) { l -> param("ln2b.$l", config.dModel) { 0f } }
    val ffn1W: Array<FloatArray> = Array(config.nLayers) { l ->
        param("ffn1W.$l", config.dModel * config.ffnHidden) { xavier(config.dModel, config.ffnHidden) }
    }
    val ffn1B: Array<FloatArray> = Array(config.nLayers) { l -> param("ffn1B.$l", config.ffnHidden) { 0f } }
    val ffn2W: Array<FloatArray> = Array(config.nLayers) { l ->
        param("ffn2W.$l", config.ffnHidden * config.dModel) { xavier(config.ffnHidden, config.dModel) }
    }
    val ffn2B: Array<FloatArray> = Array(config.nLayers) { l -> param("ffn2B.$l", config.dModel) { 0f } }

    val lnFg: FloatArray = param("lnFg", config.dModel) { 1f }
    val lnFb: FloatArray = param("lnFb", config.dModel) { 0f }
    val headW: FloatArray = param("headW", config.dModel * config.vocab) { xavier(config.dModel, config.vocab) }
    val headB: FloatArray = param("headB", config.vocab) { 0f }

    val numParams: Long get() = params.sumOf { it.data.size.toLong() }

    // ------------------------------------------------------------------
    // Worker: thread-local forward/backward caches + gradient buffers.
    // One Worker per training thread; caches are allocated once and reused.
    // ------------------------------------------------------------------
    inner class Worker {
        private val T = config.seqLen
        private val D = config.dModel
        private val L = config.nLayers
        private val H = config.nHeads
        private val Dh = config.headDim
        private val F = config.ffnHidden
        private val V = config.vocab
        private val sqrtDh = sqrt(Dh.toFloat())

        // forward caches
        val xs: Array<FloatArray> = Array(L + 1) { FloatArray(T * D) }
        val ln1: Array<FloatArray> = Array(L) { FloatArray(T * D) }
        val ln1mean: Array<FloatArray> = Array(L) { FloatArray(T) }
        val ln1inv: Array<FloatArray> = Array(L) { FloatArray(T) }
        val qkv: Array<FloatArray> = Array(L) { FloatArray(T * 3 * D) }
        val probs: Array<FloatArray> = Array(L) { FloatArray(H * T * T) }
        val avo: Array<FloatArray> = Array(L) { FloatArray(T * D) }
        val xMid: Array<FloatArray> = Array(L) { FloatArray(T * D) }
        val ln2: Array<FloatArray> = Array(L) { FloatArray(T * D) }
        val ln2mean: Array<FloatArray> = Array(L) { FloatArray(T) }
        val ln2inv: Array<FloatArray> = Array(L) { FloatArray(T) }
        val f1pre: Array<FloatArray> = Array(L) { FloatArray(T * F) }
        val f1: Array<FloatArray> = Array(L) { FloatArray(T * F) }
        val lnF = FloatArray(T * D)
        val lnFmean = FloatArray(T)
        val lnFinv = FloatArray(T)
        val logits = FloatArray(T * V)
        val outProbs = FloatArray(T * V)

        // backward scratch
        val gLogits = FloatArray(T * V)
        val gX = FloatArray(T * D)      // d(xs[l+1]) while descending
        val gXMid = FloatArray(T * D)
        val gTmp = FloatArray(T * D)
        val gF1 = FloatArray(T * F)
        val gQkv = FloatArray(T * 3 * D)
        val scratch = FloatArray(T * D) // proj / ffn2 outputs in forward

        // gradient buffers, parallel to GPT.params
        val grads: Array<FloatArray> = Array(params.size) { i -> FloatArray(params[i].data.size) }

        // param index lookup (built once)
        private val idx = HashMap<String, Int>().also {
            params.forEachIndexed { i, p -> it[p.name] = i }
        }
        fun gradOf(name: String): FloatArray = grads[idx[name]!!]

        fun zeroGrads() {
            for (g in grads) g.fill(0f)
        }

        /**
         * Forward pass on one sequence of T+1 token ids.
         * Positions 0..T-1 are inputs; targets are seq[1..T].
         * Returns mean cross-entropy loss over the T positions.
         */
        fun forward(seq: IntArray): Float {
            // embeddings
            val x0 = xs[0]
            for (t in 0 until T) {
                val tok = seq[t]
                val eOff = tok * D
                val pOff = t * D
                val oOff = t * D
                for (j in 0 until D) x0[oOff + j] = tokEmb[eOff + j] + posEmb[pOff + j]
            }
            for (l in 0 until L) {
                val xin = xs[l]
                // --- attention sub-block ---
                layerNormFwd(xin, T, D, ln1g[l], ln1b[l], ln1[l], ln1mean[l], ln1inv[l])
                mm(ln1[l], T, D, qkvW[l], 3 * D, qkv[l])
                addBias(qkv[l], T, 3 * D, qkvB[l])
                // causal multi-head attention
                val pr = probs[l]
                val q = qkv[l]
                for (h in 0 until H) {
                    val hOff = h * T * T
                    val qOff0 = h * Dh
                    val kOff0 = D + h * Dh
                    val vOff0 = 2 * D + h * Dh
                    for (i in 0 until T) {
                        val qi = i * 3 * D + qOff0
                        val pi = hOff + i * T
                        // scores for j <= i (causal: no peeking at the future)
                        for (j in 0..i) {
                            pr[pi + j] = dot(q, qi, q, j * 3 * D + kOff0, Dh) / sqrtDh
                        }
                        softmaxSlice(pr, pi, i + 1, pr, pi)
                    }
                    // weighted sum of values
                    val av = avo[l]
                    for (i in 0 until T) {
                        val pi = hOff + i * T
                        val oi = i * D + h * Dh
                        for (d in 0 until Dh) av[oi + d] = 0f
                        for (j in 0..i) {
                            val p = pr[pi + j]
                            val vj = j * 3 * D + vOff0
                            for (d in 0 until Dh) av[oi + d] += p * q[vj + d]
                        }
                    }
                }
                mm(avo[l], T, D, outW[l], D, scratch)
                addBias(scratch, T, D, outB[l])
                val xm = xMid[l]
                for (k in xin.indices) xm[k] = xin[k] + scratch[k]
                // --- MLP sub-block ---
                layerNormFwd(xm, T, D, ln2g[l], ln2b[l], ln2[l], ln2mean[l], ln2inv[l])
                mm(ln2[l], T, D, ffn1W[l], F, f1pre[l])
                addBias(f1pre[l], T, F, ffn1B[l])
                reluInto(f1pre[l], f1[l])
                mm(f1[l], T, F, ffn2W[l], D, scratch)
                addBias(scratch, T, D, ffn2B[l])
                val xout = xs[l + 1]
                for (k in xm.indices) xout[k] = xm[k] + scratch[k]
            }
            layerNormFwd(xs[L], T, D, lnFg, lnFb, lnF, lnFmean, lnFinv)
            mm(lnF, T, D, headW, V, logits)
            addBias(logits, T, V, headB)
            softmaxRows(logits, T, V, V, outProbs)
            var loss = 0f
            for (t in 0 until T) {
                val p = outProbs[t * V + seq[t + 1]].coerceAtLeast(1e-12f)
                loss -= kotlin.math.ln(p.toDouble()).toFloat()
            }
            return loss / T
        }

        /**
         * Backward pass for the sequence just run through [forward].
         * Accumulates gradients into [grads]. Call [zeroGrads] first per batch.
         */
        fun backward(seq: IntArray) {
            // d logits: (p - onehot) / T
            for (t in 0 until T) {
                val off = t * V
                for (v in 0 until V) gLogits[off + v] = outProbs[off + v]
                gLogits[off + seq[t + 1]] -= 1f
            }
            scaleInto(gLogits, 1f / T)
            // output head
            tmMAcc(lnF, T, D, gLogits, V, gradOf("headW"))
            colSumAcc(gLogits, T, V, gradOf("headB"))
            mmT(gLogits, T, V, headW, D, gX) // d(lnF)
            layerNormBwd(gX, xs[L], T, D, lnFg, lnFmean, lnFinv, gTmp, gradOf("lnFg"), gradOf("lnFb"))
            // gTmp now holds d(xs[L]); move into gX for the descent
            gTmp.copyInto(gX)

            for (l in L - 1 downTo 0) {
                // ---- MLP branch ----
                tmMAcc(f1[l], T, F, gX, D, gradOf("ffn2W.$l"))
                colSumAcc(gX, T, D, gradOf("ffn2B.$l"))
                mmT(gX, T, D, ffn2W[l], F, gF1) // d relu(f1pre)
                val fp = f1pre[l]
                for (k in gF1.indices) gF1[k] = if (fp[k] > 0f) gF1[k] else 0f
                tmMAcc(ln2[l], T, D, gF1, F, gradOf("ffn1W.$l"))
                colSumAcc(gF1, T, F, gradOf("ffn1B.$l"))
                mmT(gF1, T, F, ffn1W[l], D, gTmp) // d(ln2)
                layerNormBwd(gTmp, xMid[l], T, D, ln2g[l], ln2mean[l], ln2inv[l],
                    scratch, gradOf("ln2g.$l"), gradOf("ln2b.$l"))
                // scratch = d(xMid) from MLP branch; add residual passthrough gX
                for (k in gXMid.indices) gXMid[k] = gX[k] + scratch[k]

                // ---- attention branch ----
                tmMAcc(avo[l], T, D, gXMid, D, gradOf("outW.$l"))
                colSumAcc(gXMid, T, D, gradOf("outB.$l"))
                mmT(gXMid, T, D, outW[l], D, gTmp) // d(avo)
                // attention backward -> gQkv
                gQkv.fill(0f)
                val pr = probs[l]
                val q = qkv[l]
                for (h in 0 until H) {
                    val hOff = h * T * T
                    val qOff0 = h * Dh
                    val kOff0 = D + h * Dh
                    val vOff0 = 2 * D + h * Dh
                    for (i in 0 until T) {
                        val pi = hOff + i * T
                        val oi = i * D + h * Dh // dO row lives in gTmp (= d(avo))
                        // dp[j] = dO_i . V_j ; accumulate dV_j += p_ij * dO_i
                        for (j in 0..i) {
                            val p = pr[pi + j]
                            val vj = j * 3 * D + vOff0
                            var dpj = 0f
                            var d = 0
                            while (d < Dh) {
                                dpj += gTmp[oi + d] * q[vj + d]
                                gQkv[vj + d] += p * gTmp[oi + d]
                                d++
                            }
                            dpBuf[j] = dpj
                        }
                        // softmax backward over the causal row
                        var rowDot = 0f
                        for (j in 0..i) rowDot += pr[pi + j] * dpBuf[j]
                        for (j in 0..i) {
                            val ds = pr[pi + j] * (dpBuf[j] - rowDot) / sqrtDh
                            val qi = i * 3 * D + qOff0
                            val kj = j * 3 * D + kOff0
                            for (d in 0 until Dh) {
                                gQkv[qi + d] += ds * q[kj + d] // dQ_i += ds * K_j
                                gQkv[kj + d] += ds * q[qi + d] // dK_j += ds * Q_i
                            }
                        }
                    }
                }
                tmMAcc(ln1[l], T, D, gQkv, 3 * D, gradOf("qkvW.$l"))
                colSumAcc(gQkv, T, 3 * D, gradOf("qkvB.$l"))
                mmT(gQkv, T, 3 * D, qkvW[l], D, gTmp) // d(ln1)
                layerNormBwd(gTmp, xs[l], T, D, ln1g[l], ln1mean[l], ln1inv[l],
                    scratch, gradOf("ln1g.$l"), gradOf("ln1b.$l"))
                // gX = d(xs[l]) = residual passthrough (gXMid) + attn branch (scratch)
                for (k in gX.indices) gX[k] = gXMid[k] + scratch[k]
            }
            // embeddings
            val gTok = gradOf("tokEmb")
            val gPos = gradOf("posEmb")
            for (t in 0 until T) {
                val tok = seq[t]
                val eOff = tok * D
                val pOff = t * D
                for (j in 0 until D) {
                    val g = gX[t * D + j]
                    gTok[eOff + j] += g
                    gPos[pOff + j] += g
                }
            }
        }

        /** Small per-row buffer for attention softmax backward. Sized T. */
        private val dpBuf = FloatArray(config.seqLen)

        /** Forward pass returning only the last-position logits (for generation). */
        fun forwardLogitsLast(context: IntArray): FloatArray {
            // context length n <= T; run the full forward on a padded copy, read last row
            val seq = IntArray(T + 1)
            val n = context.size.coerceAtMost(T)
            val start = T - n
            for (t in 0 until T) seq[t] = if (t < start) Tokenizer.PAD else context[t - start]
            forward(seq) // fills logits; PAD rows produce garbage but we only read row T-1
            return logits.copyOfRange((T - 1) * V, T * V)
        }
    }

    fun newWorker(): Worker = Worker()

    // ---------------- persistence ----------------
    fun save(file: File) {
        DataOutputStream(FileOutputStream(file).buffered()).use { d ->
            d.writeInt(0x46475054) // "FGPT"
            d.writeInt(1) // version
            d.writeInt(config.vocab); d.writeInt(config.dModel); d.writeInt(config.nLayers)
            d.writeInt(config.nHeads); d.writeInt(config.seqLen); d.writeInt(config.ffnHidden)
            d.writeInt(params.size)
            for (p in params) {
                d.writeUTF(p.name)
                d.writeInt(p.data.size)
                for (v in p.data) d.writeFloat(v)
            }
        }
    }

    companion object {
        fun load(file: File): GPT {
            DataInputStream(FileInputStream(file).buffered()).use { d ->
                require(d.readInt() == 0x46475054) { "not a ForgeGPT weights file" }
                require(d.readInt() == 1) { "unsupported weights version" }
                val cfg = GPTConfig(
                    vocab = d.readInt(), dModel = d.readInt(), nLayers = d.readInt(),
                    nHeads = d.readInt(), seqLen = d.readInt(), ffnHidden = d.readInt()
                )
                val model = GPT(cfg)
                val n = d.readInt()
                require(n == model.params.size) { "param count mismatch: $n vs ${model.params.size}" }
                for (i in 0 until n) {
                    val name = d.readUTF()
                    val size = d.readInt()
                    val p = model.params[i]
                    require(p.name == name && p.data.size == size) { "param mismatch at $i: $name" }
                    for (j in p.data.indices) p.data[j] = d.readFloat()
                }
                return model
            }
        }
    }
}
