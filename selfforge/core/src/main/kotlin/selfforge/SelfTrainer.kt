package selfforge

/** What happened in one round of self-training. */
data class RoundReport(
    val round: Int,
    val trainAccuracy: Double,
    val valAccuracy: Double,
    val pseudoLabeledAdded: Int,
    val poolRemaining: Int
)

/**
 * The recursive self-teacher.
 *
 * This is the real "self-training" algorithm from machine learning:
 * the model labels unlabeled data itself, keeps only the predictions it is
 * confident about, adds them to its training set as if they were ground
 * truth, and retrains. Each round's improved model teaches the next round —
 * the model recursively improves itself without any new human labels.
 *
 * Two guardrails keep it honest:
 * 1. Consistency filter — a self-made label is only trusted if the model
 *    predicted the same label for that input in the previous round too.
 *    Flip-flopping boundary guesses are never trusted.
 * 2. Best-snapshot restore — the weights from the best validation round are
 *    kept, and restored at the end. Self-training can never finish worse
 *    than it started (on the held-out set).
 */
class SelfTrainer(
    /** Minimum confidence (0..1) for a self-made label to be trusted. */
    val confidenceThreshold: Double = 0.75,
    /** Hard cap on self-training rounds. */
    val maxRounds: Int = 12,
    /** Epochs of gradient descent per round. */
    val epochsPerRound: Int = 400,
    /** Stop early if validation accuracy doesn't improve for this many rounds. */
    val patience: Int = 4,
    /** Require the same self-label two rounds in a row before trusting it. */
    val requireConsistentLabel: Boolean = true
) {
    private class PoolItem(val x: DoubleArray, var lastLabel: Int? = null)

    /**
     * Run the recursive loop.
     * @param net the model to improve (its weights are mutated in place)
     * @param labeled human-provided examples
     * @param unlabeledPool raw inputs with no labels — the model labels these itself
     * @param validation held-out examples used to judge whether rounds actually help
     */
    fun selfTrain(
        net: NeuralNet,
        labeled: List<Example>,
        unlabeledPool: List<DoubleArray>,
        validation: List<Example>
    ): List<RoundReport> {
        val pool = unlabeledPool.map { PoolItem(it) }.toMutableList()
        val training = labeled.toMutableList()
        val reports = mutableListOf<RoundReport>()

        var bestVal = net.accuracy(validation)
        var bestWeights = net.exportWeights()
        var staleRounds = 0
        var emptyRounds = 0

        for (round in 1..maxRounds) {
            net.trainBatch(training, epochsPerRound)
            val trainAcc = net.accuracy(training)
            val valAcc = net.accuracy(validation)

            if (valAcc > bestVal + 1e-9) {
                bestVal = valAcc
                bestWeights = net.exportWeights()
                staleRounds = 0
            } else {
                staleRounds++
            }

            // The model teaches itself: label the pool, keep confident AND
            // consistent ones. Everything else stays in the pool for next round.
            // (Round 1 only records predictions — consistency needs two rounds.)
            val newlyLabeled = mutableListOf<Example>()
            val it = pool.iterator()
            while (it.hasNext()) {
                val item = it.next()
                val probs = net.predictProbs(item.x)
                var best = 0
                for (j in 1 until probs.size) if (probs[j] > probs[best]) best = j
                val consistent = !requireConsistentLabel || item.lastLabel == best
                if (probs[best] >= confidenceThreshold && consistent) {
                    newlyLabeled.add(Example(item.x, best))
                    it.remove()
                } else {
                    item.lastLabel = best
                }
            }
            training.addAll(newlyLabeled)
            reports.add(RoundReport(round, trainAcc, valAcc, newlyLabeled.size, pool.size))

            if (newlyLabeled.isEmpty()) emptyRounds++ else emptyRounds = 0
            if (emptyRounds >= 2 || staleRounds >= patience) break
        }

        // Restore the best the model ever was — never finish worse than baseline.
        net.copyWeightsFrom(NeuralNet.importWeights(bestWeights, net.learningRate))
        return reports
    }
}
