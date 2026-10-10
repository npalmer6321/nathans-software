package forgegpt

import kotlin.math.sqrt

/**
 * Flat-array linear algebra, row-major. Everything is FloatArray so the JVM
 * JIT can vectorize the tight loops. No external dependencies.
 *
 * Naming: mm = C = A @ B, mmT = C = A @ B^T, tmM = C = A^T @ B.
 * All "Into/accumulate" variants take a preallocated out array.
 */

/** out[ar,bc] = a[ar,ac] @ b[ac,bc]. out is overwritten. */
fun mm(a: FloatArray, ar: Int, ac: Int, b: FloatArray, bc: Int, out: FloatArray) {
    out.fill(0f)
    var i = 0
    while (i < ar) {
        val aOff = i * ac
        val oOff = i * bc
        var k = 0
        while (k < ac) {
            val aik = a[aOff + k]
            val bOff = k * bc
            var j = 0
            while (j < bc) {
                out[oOff + j] += aik * b[bOff + j]
                j++
            }
            k++
        }
        i++
    }
}

/** out[ar,br] = a[ar,ac] @ b[br,ac]^T. out is overwritten. */
fun mmT(a: FloatArray, ar: Int, ac: Int, b: FloatArray, br: Int, out: FloatArray) {
    var i = 0
    while (i < ar) {
        val aOff = i * ac
        val oOff = i * br
        var j = 0
        while (j < br) {
            val bOff = j * ac
            var s = 0f
            var k = 0
            while (k < ac) {
                s += a[aOff + k] * b[bOff + k]
                k++
            }
            out[oOff + j] = s
            j++
        }
        i++
    }
}

/** out[ac,br] = a[ar,ac]^T @ b[ar,br]. out is overwritten. */
fun tmM(a: FloatArray, ar: Int, ac: Int, b: FloatArray, br: Int, out: FloatArray) {
    out.fill(0f)
    var i = 0
    while (i < ar) {
        val aOff = i * ac
        val bOff = i * br
        var k = 0
        while (k < ac) {
            val aik = a[aOff + k]
            val oOff = k * br
            var j = 0
            while (j < br) {
                out[oOff + j] += aik * b[bOff + j]
                j++
            }
            k++
        }
        i++
    }
}

/** y[n,cols] += bias[cols] */
fun addBias(y: FloatArray, n: Int, cols: Int, bias: FloatArray) {
    var i = 0
    while (i < n) {
        val off = i * cols
        var j = 0
        while (j < cols) {
            y[off + j] += bias[j]
            j++
        }
        i++
    }
}

/** sums[n] = column sums of m[rows, n]; sums is overwritten. */
fun colSumInto(m: FloatArray, rows: Int, n: Int, sums: FloatArray) {
    sums.fill(0f)
    colSumAcc(m, rows, n, sums)
}

/** sums[n] += column sums of m[rows, n]. */
fun colSumAcc(m: FloatArray, rows: Int, n: Int, sums: FloatArray) {
    var i = 0
    while (i < rows) {
        val off = i * n
        var j = 0
        while (j < n) {
            sums[j] += m[off + j]
            j++
        }
        i++
    }
}

/** out[ac,br] += a[ar,ac]^T @ b[ar,br]. out is NOT zeroed (accumulating). */
fun tmMAcc(a: FloatArray, ar: Int, ac: Int, b: FloatArray, br: Int, out: FloatArray) {
    var i = 0
    while (i < ar) {
        val aOff = i * ac
        val bOff = i * br
        var k = 0
        while (k < ac) {
            val aik = a[aOff + k]
            val oOff = k * br
            var j = 0
            while (j < br) {
                out[oOff + j] += aik * b[bOff + j]
                j++
            }
            k++
        }
        i++
    }
}

/** out = relu(x) elementwise. */
fun reluInto(x: FloatArray, out: FloatArray) {
    for (i in x.indices) out[i] = if (x[i] > 0f) x[i] else 0f
}

/** dst += src elementwise. */
fun addInto(dst: FloatArray, src: FloatArray) {
    for (i in dst.indices) dst[i] += src[i]
}

/** dst *= s. */
fun scaleInto(dst: FloatArray, s: Float) {
    for (i in dst.indices) dst[i] *= s
}

fun dot(a: FloatArray, aOff: Int, b: FloatArray, bOff: Int, n: Int): Float {
    var s = 0f
    var k = 0
    while (k < n) {
        s += a[aOff + k] * b[bOff + k]
        k++
    }
    return s
}

/**
 * Row-wise softmax over the first [len] entries of each row.
 * rows: number of rows, stride: full row width, out may alias x.
 */
fun softmaxRows(x: FloatArray, rows: Int, stride: Int, len: Int, out: FloatArray) {
    var i = 0
    while (i < rows) {
        val off = i * stride
        var m = x[off]
        var j = 1
        while (j < len) {
            if (x[off + j] > m) m = x[off + j]
            j++
        }
        var s = 0f
        j = 0
        while (j < len) {
            val e = kotlin.math.exp((x[off + j] - m).toDouble()).toFloat()
            out[off + j] = e
            s += e
            j++
        }
        val inv = 1f / s
        j = 0
        while (j < len) {
            out[off + j] *= inv
            j++
        }
        i++
    }
}

/** Softmax over x[off, off+len), written in place or to out[outOff, ...). */
fun softmaxSlice(x: FloatArray, off: Int, len: Int, out: FloatArray, outOff: Int) {
    var m = x[off]
    var j = 1
    while (j < len) {
        if (x[off + j] > m) m = x[off + j]
        j++
    }
    var s = 0f
    j = 0
    while (j < len) {
        val e = kotlin.math.exp((x[off + j] - m).toDouble()).toFloat()
        out[outOff + j] = e
        s += e
        j++
    }
    val inv = 1f / s
    j = 0
    while (j < len) {
        out[outOff + j] *= inv
        j++
    }
}

/**
 * LayerNorm forward, per row: out = (x - mean)/std * g + b.
 * mean/invStd are per-row caches for the backward pass.
 */
fun layerNormFwd(
    x: FloatArray, n: Int, d: Int,
    g: FloatArray, b: FloatArray,
    out: FloatArray, mean: FloatArray, invStd: FloatArray
) {
    var i = 0
    while (i < n) {
        val off = i * d
        var s = 0f
        var j = 0
        while (j < d) {
            s += x[off + j]
            j++
        }
        val mu = s / d
        var v = 0f
        j = 0
        while (j < d) {
            val dd = x[off + j] - mu
            v += dd * dd
            j++
        }
        val inv = 1f / sqrt(v / d + 1e-5f)
        mean[i] = mu
        invStd[i] = inv
        j = 0
        while (j < d) {
            out[off + j] = (x[off + j] - mu) * inv * g[j] + b[j]
            j++
        }
        i++
    }
}

/**
 * LayerNorm backward. Accumulates dg/db; writes dx (overwritten).
 */
fun layerNormBwd(
    dout: FloatArray, x: FloatArray,
    n: Int, d: Int, g: FloatArray,
    mean: FloatArray, invStd: FloatArray,
    dx: FloatArray, dg: FloatArray, db: FloatArray
) {
    var i = 0
    while (i < n) {
        val off = i * d
        val inv = invStd[i]
        var s1 = 0f
        var s2 = 0f
        var j = 0
        while (j < d) {
            val dg_j = dout[off + j] * g[j]
            s1 += dg_j
            // xhat: normalized input before gamma/beta
            val xhat = (x[off + j] - mean[i]) * inv
            s2 += dg_j * xhat
            j++
        }
        j = 0
        while (j < d) {
            val xhat = (x[off + j] - mean[i]) * inv
            dx[off + j] = inv / d * (d * dout[off + j] * g[j] - s1 - xhat * s2)
            dg[j] += dout[off + j] * xhat
            db[j] += dout[off + j]
            j++
        }
        i++
    }
}
