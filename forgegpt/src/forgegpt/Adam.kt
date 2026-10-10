package forgegpt

import kotlin.math.sqrt

/**
 * Adam optimizer (Kingma & Ba 2014), hand-written.
 * Keeps first/second moment estimates per parameter.
 */
class Adam(
    private val model: GPT,
    private var lr: Float = 3e-4f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val eps: Float = 1e-8f,
    private val maxGradNorm: Float = 1.0f
) {
    private val m: Array<FloatArray> = Array(model.params.size) { i -> FloatArray(model.params[i].data.size) }
    private val v: Array<FloatArray> = Array(model.params.size) { i -> FloatArray(model.params[i].data.size) }
    private var step = 0

    fun setLr(lr: Float) {
        this.lr = lr
    }

    /** One optimizer step given mean gradients (parallel to model.params). */
    fun step(grads: Array<FloatArray>) {
        step++
        // global grad-norm clipping
        var normSq = 0.0
        for (g in grads) for (x in g) normSq += x.toDouble() * x.toDouble()
        val norm = sqrt(normSq).toFloat()
        val clipScale = if (norm > maxGradNorm) maxGradNorm / norm else 1f

        val b1t = 1f - Math.pow(beta1.toDouble(), step.toDouble()).toFloat()
        val b2t = 1f - Math.pow(beta2.toDouble(), step.toDouble()).toFloat()
        for (i in model.params.indices) {
            val p = model.params[i].data
            val g = grads[i]
            val mi = m[i]
            val vi = v[i]
            for (j in p.indices) {
                val gj = g[j] * clipScale
                mi[j] = beta1 * mi[j] + (1f - beta1) * gj
                vi[j] = beta2 * vi[j] + (1f - beta2) * gj * gj
                val mHat = mi[j] / b1t
                val vHat = vi[j] / b2t
                p[j] -= lr * mHat / (sqrt(vHat) + eps)
            }
        }
    }

    fun gradNorm(grads: Array<FloatArray>): Float {
        var s = 0.0
        for (g in grads) for (x in g) s += x.toDouble() * x.toDouble()
        return sqrt(s).toFloat()
    }
}
