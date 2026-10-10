package selfforge.codegen

import kotlin.random.Random

/**
 * Proves the code model works: trains a char-RNN on the corpus, then runs
 * the recursive self-teaching loop (generate -> run -> keep -> retrain) and
 * shows the share of valid programs per round, with real samples.
 */
fun main() {
    val vocab = CodeCorpus.vocab
    println("vocab (${vocab.size} chars): ${vocab.joinToString("").replace("\n", "\\n")}")
    val rnn = CharRNN(vocab, hiddenSize = 96, learningRate = 0.05, seed = 7L)
    println("code model weights: ${rnn.weightCount()}")

    // Warmup: learn the shape of code from the human corpus.
    rnn.train(CodeCorpus.programs, epochs = 250)
    println("train loss after warmup: ${"%.3f".format(rnn.loss(CodeCorpus.programs))}")

    val interp = TinyPy()
    val rng = Random(1234L)

    fun validPct(n: Int): Double {
        var valid = 0
        repeat(n) {
            val seedLine = CodeCorpus.programs[rng.nextInt(CodeCorpus.programs.size)].lineSequence().first()
            val program = rnn.sample(seedLine + "\n", 140, 0.9, rng).trim() + "\n"
            if (interp.run(program) is TinyPy.Result.Ok) valid++
        }
        return valid.toDouble() / n
    }

    val baseline = validPct(60)
    println("\n[baseline] valid programs before self-teaching: ${"%.0f".format(baseline * 100)}%")

    val corpus = CodeCorpus.programs.toMutableList()
    val trainer = CodeSelfTrainer(programsPerRound = 40, maxRounds = 6, epochsPerRound = 60, temperature = 0.9)
    val reports = trainer.selfTeach(rnn, corpus, interp, seed = 99L)
    println("\n[self-teaching rounds]")
    for (r in reports) {
        println("  round ${r.round}: valid=${"%.0f".format(r.validPct * 100)}% " +
                "kept ${r.kept} (training set now ${r.corpusSize})")
    }
    val final = validPct(60)
    println("\n[result] valid programs: ${"%.0f".format(baseline * 100)}% -> ${"%.0f".format(final * 100)}%")

    val sample = reports.lastOrNull()?.sampleProgram
    if (!sample.isNullOrEmpty()) {
        println("\n[a program it wrote itself]")
        println(sample)
        println("[running it]")
        when (val res = interp.run(sample)) {
            is TinyPy.Result.Ok -> println("output: ${res.output}")
            is TinyPy.Result.Err -> println("error: ${res.message}")
        }
    }
}
