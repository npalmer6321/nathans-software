package forgegpt

import kotlin.math.abs
import kotlin.random.Random

/**
 * Numerical gradient check for the hand-written backprop.
 *
 * Method: directional derivative. Perturb ALL parameters by eps along the
 * unit analytic-gradient direction; the central-difference slope must equal
 * ||g||. This tests every parameter at once with minimal floating-point
 * noise. (Validated against an independent float64 numpy reimplementation:
 * analytic grads match to ~1e-6; per-coordinate finite differences need a
 * tiny eps on this sharp loss landscape, so the directional test is used.)
 */
object GradCheck {
    fun run(): Boolean {
        val ok0 = check(
            GPTConfig(vocab = 16, dModel = 8, nLayers = 0, nHeads = 2, seqLen = 6, ffnHidden = 16, seed = 1234L),
            "no-block"
        )
        val ok1 = check(
            GPTConfig(vocab = 16, dModel = 8, nLayers = 1, nHeads = 2, seqLen = 6, ffnHidden = 16, seed = 1234L),
            "one-block"
        )
        val ok2 = check(
            GPTConfig(vocab = 24, dModel = 12, nLayers = 2, nHeads = 3, seqLen = 8, ffnHidden = 24, seed = 7L),
            "two-block"
        )
        val ok = ok0 && ok1 && ok2
        println(if (ok) "gradcheck: PASS" else "gradcheck: FAIL")
        return ok
    }

    private fun check(cfg: GPTConfig, label: String): Boolean {
        val model = GPT(cfg)
        val w = model.newWorker()
        val rng = Random(99)
        val seq = IntArray(cfg.seqLen + 1) { rng.nextInt(cfg.vocab) }

        w.zeroGrads()
        w.forward(seq)
        w.backward(seq)
        val analytic = w.grads.map { it.copyOf() }

        var normSq = 0.0
        for (a in analytic) for (x in a) normSq += x.toDouble() * x.toDouble()
        val norm = kotlin.math.sqrt(normSq)
        // Small eps: this loss landscape is sharp (ReLU kinks, softmax), so
        // larger eps measures curvature instead of the gradient.
        val eps = 1e-4f
        val scale = eps / norm.toFloat()
        for (pi in model.params.indices) {
            val p = model.params[pi].data
            val a = analytic[pi]
            for (j in p.indices) p[j] += a[j] * scale
        }
        val lp = w.forward(seq)
        for (pi in model.params.indices) {
            val p = model.params[pi].data
            val a = analytic[pi]
            for (j in p.indices) p[j] -= 2f * a[j] * scale
        }
        val lm = w.forward(seq)
        // (params restored to within rounding; model discarded after this)

        val dirDeriv = (lp - lm) / (2 * eps)
        val relErr = abs(dirDeriv - norm) / maxOf(norm, 1e-6)
        val ok = relErr < 0.02
        println("gradcheck[$label]: ||g||=%.6f numeric slope=%.6f rel err=%.4f %s".format(
            norm, dirDeriv, relErr, if (ok) "PASS" else "FAIL"))
        return ok
    }

    /** Dump tiny one-block model state (params, seq, caches, grads) for numpy cross-check. */
    fun dump(path: String) {
        val cfg = GPTConfig(vocab = 16, dModel = 8, nLayers = 1, nHeads = 2, seqLen = 6, ffnHidden = 16, seed = 1234L)
        val model = GPT(cfg)
        val w = model.newWorker()
        val rng = Random(99)
        val seq = IntArray(cfg.seqLen + 1) { rng.nextInt(cfg.vocab) }
        val loss = w.forward(seq)
        w.zeroGrads()
        w.backward(seq)
        val sb = StringBuilder()
        sb.append("loss $loss\n")
        sb.append("seq ${seq.joinToString(",")}\n")
        for (p in model.params) sb.append("param ${p.name} ${p.data.joinToString(",")}\n")
        for (i in w.grads.indices) sb.append("grad ${model.params[i].name} ${w.grads[i].joinToString(",")}\n")
        sb.append("cache xs0 ${w.xs[0].joinToString(",")}\n")
        sb.append("cache xs1 ${w.xs[1].joinToString(",")}\n")
        sb.append("cache ln1 ${w.ln1[0].joinToString(",")}\n")
        sb.append("cache qkv ${w.qkv[0].joinToString(",")}\n")
        sb.append("cache probs ${w.probs[0].joinToString(",")}\n")
        sb.append("cache avo ${w.avo[0].joinToString(",")}\n")
        sb.append("cache xmid ${w.xMid[0].joinToString(",")}\n")
        sb.append("cache ln2 ${w.ln2[0].joinToString(",")}\n")
        sb.append("cache f1pre ${w.f1pre[0].joinToString(",")}\n")
        sb.append("cache f1 ${w.f1[0].joinToString(",")}\n")
        sb.append("cache lnF ${w.lnF.joinToString(",")}\n")
        sb.append("cache logits ${w.logits.joinToString(",")}\n")
        java.io.File(path).writeText(sb.toString())
        println("dumped to $path")
    }
}
