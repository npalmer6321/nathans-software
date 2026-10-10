package selfforge

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * End-to-end proof that the self-training core works.
 *
 * Task: 2D points drawn from 4 overlapping Gaussian blobs in an XOR layout
 * (2 blobs = class A, 2 blobs = class B). The human teaches only 8 examples
 * (2 per blob). The model then recursively teaches ITSELF from 600 unlabeled
 * points: each round it labels the points it is confident AND consistent
 * about, adds them to its training set, and retrains. Judged on 400 held-out
 * points it never trains on.
 */
fun main() {
    val rng = Random(1234L)
    val blobs = arrayOf(
        doubleArrayOf(-0.55, -0.55) to 0, doubleArrayOf(0.55, 0.55) to 0,
        doubleArrayOf(-0.55, 0.55) to 1, doubleArrayOf(0.55, -0.55) to 1
    )
    fun sampleBlob(b: Int): Pair<DoubleArray, Int> {
        val (c, cls) = blobs[b]
        return doubleArrayOf(c[0] + rng.nextGaussian() * 0.28, c[1] + rng.nextGaussian() * 0.28) to cls
    }

    val labeled = mutableListOf<Example>()
    for (b in 0..3) repeat(2) { val (p, c) = sampleBlob(b); labeled.add(Example(p, c)) }
    val unlabeled = (1..600).map { sampleBlob(rng.nextInt(4)).first }
    val validation = (1..400).map { val (p, c) = sampleBlob(rng.nextInt(4)); Example(p, c) }

    val registry = SkillRegistry()
    val skill = registry.newSkill("xor-blobs", 2, listOf("class-a", "class-b"), hiddenSize = 24, seed = 7L)
    skill.net.learningRate = 0.1
    for (e in labeled) skill.teach(e.input, skill.labels[e.label])

    println("model weights: ${skill.net.weightCount()}")
    println("human-taught examples: ${labeled.size}")
    println("unlabeled examples for self-teaching: ${unlabeled.size}")
    println("held-out validation examples: ${validation.size}")

    // Baseline: supervised training only, no self-teaching.
    skill.train(epochs = 1500)
    val baseline = skill.accuracy(validation)
    println("\n[baseline] supervised-only validation accuracy: ${"%.1f".format(baseline * 100)}%")

    // Recursive self-training: the model is now its own teacher.
    val reports = skill.selfTrain(
        unlabeled, validation,
        SelfTrainer(confidenceThreshold = 0.80, maxRounds = 12, epochsPerRound = 400, patience = 4)
    )
    println("\n[self-training rounds]")
    for (r in reports) {
        println("  round ${r.round}: val=${"%.1f".format(r.valAccuracy * 100)}% " +
                "self-labeled +${r.pseudoLabeledAdded} (pool left ${r.poolRemaining})")
    }
    val final = skill.accuracy(validation)
    val gain = (final - baseline) * 100
    println("\n[result] validation accuracy after self-training: ${"%.1f".format(final * 100)}%")
    println("[result] improvement from recursive self-teaching: ${"%+.1f".format(gain)} points")

    // A second, independent skill proves the "new abilities" part.
    val circle = registry.newSkill("inside-circle", 2, listOf("inside", "outside"), hiddenSize = 24, seed = 11L)
    circle.net.learningRate = 0.1
    repeat(24) {
        val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
        circle.teach(p, if (sqrt(p[0] * p[0] + p[1] * p[1]) < 0.55) "inside" else "outside")
    }
    circle.train(1500)
    val circleTest = (1..300).map {
        val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
        Example(p, if (sqrt(p[0] * p[0] + p[1] * p[1]) < 0.55) 0 else 1)
    }
    println("\n[second skill] 'inside-circle' test accuracy: ${"%.1f".format(circle.accuracy(circleTest) * 100)}%")
    println("[registry] skills known: ${registry.names()}")

    // Save/load round-trip: weights must survive intact.
    val saved = skill.save()
    val restored = Skill.load(saved)
    val probe = doubleArrayOf(0.55, 0.55)
    val (g1, c1) = skill.guessWithConfidence(probe)
    val (g2, c2) = restored.guessWithConfidence(probe)
    println("\n[save/load] before: $g1 (${"%.4f".format(c1)}), after: $g2 (${"%.4f".format(c2)})")
    check(g1 == g2 && kotlin.math.abs(c1 - c2) < 1e-9) { "save/load round-trip changed the model!" }

    // Live demo of the finished skill.
    println("\n[live] (-0.55, -0.55) -> ${skill.guessWithConfidence(doubleArrayOf(-0.55, -0.55))}")
    println("[live] (0.55, -0.55) -> ${skill.guessWithConfidence(doubleArrayOf(0.55, -0.55))}")

    println("\nOK: taught with ${labeled.size} examples, then recursively taught itself " +
            "from ${600 - reports.last().poolRemaining} self-labeled examples.")
}

private fun Random.nextGaussian(): Double {
    var u = 0.0
    var v = 0.0
    while (u == 0.0) u = nextDouble()
    while (v == 0.0) v = nextDouble()
    return sqrt(-2.0 * ln(u)) * cos(2.0 * PI * v)
}
