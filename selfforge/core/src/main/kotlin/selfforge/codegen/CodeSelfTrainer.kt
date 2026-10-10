package selfforge.codegen

import kotlin.random.Random

/** What happened in one round of the model teaching itself to code. */
data class CodeRoundReport(
    val round: Int,
    val validPct: Double,
    val kept: Int,
    val corpusSize: Int,
    val sampleProgram: String,
    val sampleOutput: String
)

/**
 * The recursive code-teacher.
 *
 * Each round:
 *  1. GENERATE — the model writes fresh programs, character by character.
 *  2. RUN — every program is executed by the TinyPy interpreter.
 *  3. KEEP — programs that run without errors (and aren't copies of known
 *     ones) are added to the training corpus as if a human wrote them.
 *  4. RETRAIN — the model trains on its own successful programs mixed with
 *     the original human corpus, so it never forgets the basics.
 *
 * The next round's generator is the improved model — recursive self-teaching.
 * Metric: the share of generated programs that actually run (% valid).
 */
class CodeSelfTrainer(
    val programsPerRound: Int = 40,
    val maxRounds: Int = 6,
    val epochsPerRound: Int = 60,
    val temperature: Double = 0.9,
    val patience: Int = 3
) {
    fun selfTeach(
        rnn: CharRNN,
        corpus: MutableList<String>,
        interp: TinyPy = TinyPy(),
        seed: Long = 99L
    ): List<CodeRoundReport> {
        val rng = Random(seed)
        val known = corpus.map { it.trim() }.toMutableSet()
        val reports = mutableListOf<CodeRoundReport>()
        var bestValid = -1.0
        var stale = 0
        var lastSample = ""
        var lastOutput = ""

        for (round in 1..maxRounds) {
            var valid = 0
            val kept = mutableListOf<String>()
            repeat(programsPerRound) {
                // Seed with a random first line from the corpus so programs start sanely.
                val seedLine = corpus[rng.nextInt(corpus.size)].lineSequence().first()
                val program = rnn.sample(seedLine + "\n", 140, temperature, rng).trim() + "\n"
                when (val r = interp.run(program)) {
                    is TinyPy.Result.Ok -> {
                        valid++
                        val norm = program.trim()
                        if (norm !in known && r.output.isNotEmpty()) {
                            known.add(norm)
                            kept.add(program)
                            lastSample = program.trim()
                            lastOutput = r.output.joinToString(", ")
                        }
                    }
                    is TinyPy.Result.Err -> Unit
                }
            }
            val validPct = valid.toDouble() / programsPerRound

            // Retrain: mostly its own successes, anchored by human examples.
            val anchor = corpus.shuffled(rng).take(15)
            rnn.train(kept + anchor, epochsPerRound)
            corpus.addAll(kept)

            reports.add(CodeRoundReport(round, validPct, kept.size, corpus.size, lastSample, lastOutput))
            if (validPct > bestValid + 1e-9) { bestValid = validPct; stale = 0 } else stale++
            if (kept.isEmpty() || stale >= patience) break
        }
        return reports
    }
}
