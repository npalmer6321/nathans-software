package forgegpt

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.random.Random

/**
 * Mini-batch next-token training loop.
 *
 * Each step: sample [batch] random (seqLen+1)-token windows from the corpus,
 * run forward+backward in parallel across worker threads, average gradients,
 * take one Adam step. Logs loss; prints a generated sample every so often.
 */
object Train {
    data class Args(
        val corpus: String = "corpus/corpus.txt",
        val out: String = "model.fg",
        val vocabOut: String = "vocab.txt",
        val steps: Int = 600,
        val batch: Int = 8,
        val lr: Float = 3e-4f,
        val dModel: Int = 64,
        val nLayers: Int = 3,
        val nHeads: Int = 4,
        val seqLen: Int = 48,
        val ffnHidden: Int = 256,
        val maxVocab: Int = 2500,
        val seed: Long = 42L,
        val logEvery: Int = 25,
        val sampleEvery: Int = 200,
        val checkpointEvery: Int = 0,
        val gradCheck: Boolean = true,
        val initModel: String = ""
    )

    fun parse(argv: List<String>): Args {
        var a = Args()
        val it = argv.iterator()
        while (it.hasNext()) {
            when (val k = it.next()) {
                "--corpus" -> a = a.copy(corpus = it.next())
                "--out" -> a = a.copy(out = it.next())
                "--vocab-out" -> a = a.copy(vocabOut = it.next())
                "--steps" -> a = a.copy(steps = it.next().toInt())
                "--batch" -> a = a.copy(batch = it.next().toInt())
                "--lr" -> a = a.copy(lr = it.next().toFloat())
                "--d-model" -> a = a.copy(dModel = it.next().toInt())
                "--layers" -> a = a.copy(nLayers = it.next().toInt())
                "--heads" -> a = a.copy(nHeads = it.next().toInt())
                "--seq-len" -> a = a.copy(seqLen = it.next().toInt())
                "--ffn" -> a = a.copy(ffnHidden = it.next().toInt())
                "--vocab" -> a = a.copy(maxVocab = it.next().toInt())
                "--seed" -> a = a.copy(seed = it.next().toLong())
                "--log-every" -> a = a.copy(logEvery = it.next().toInt())
                "--sample-every" -> a = a.copy(sampleEvery = it.next().toInt())
                "--checkpoint-every" -> a = a.copy(checkpointEvery = it.next().toInt())
                "--no-gradcheck" -> a = a.copy(gradCheck = false)
                "--init-model" -> a = a.copy(initModel = it.next())
                else -> println("train: unknown arg $k (ignored)")
            }
        }
        return a
    }

    fun run(args: Args) {
        if (args.gradCheck) {
            println("Running gradient check first (verifies the hand-written backprop)...")
            if (!GradCheck.run()) {
                println("Gradient check FAILED - refusing to train on broken math.")
                return
            }
        }

        println("Loading corpus: ${args.corpus}")
        val text = File(args.corpus).readText()
        println("Building vocabulary (max ${args.maxVocab})...")
        val tok = Tokenizer.buildFromCorpus(text, args.maxVocab)
        println("Vocab size: ${tok.vocabSize}")
        println("Tokenizing corpus...")
        val stream = Tokenizer.tokenizeText(text).map { tok.tokenToId[it] ?: Tokenizer.UNK }.toIntArray()
        println("Corpus tokens: ${stream.size}")
        require(stream.size > args.seqLen + 1) { "corpus too small" }

        val cfg = GPTConfig(
            vocab = tok.vocabSize, dModel = args.dModel, nLayers = args.nLayers,
            nHeads = args.nHeads, seqLen = args.seqLen, ffnHidden = args.ffnHidden,
            seed = args.seed
        )
        val model = if (args.initModel.isNotEmpty()) {
            println("Resuming from weights: ${args.initModel}")
            val m = GPT.load(File(args.initModel))
            require(m.config.vocab == cfg.vocab && m.config.dModel == cfg.dModel &&
                    m.config.nLayers == cfg.nLayers && m.config.seqLen == cfg.seqLen &&
                    m.config.ffnHidden == cfg.ffnHidden && m.config.nHeads == cfg.nHeads) {
                "init-model config does not match training config"
            }
            m
        } else {
            GPT(cfg)
        }
        println("Parameters: ${"%,d".format(model.numParams)}")
        val adam = Adam(model, lr = args.lr)

        val threads = maxOf(1, minOf(4, Runtime.getRuntime().availableProcessors()))
        println("Worker threads: $threads")
        val pool = Executors.newFixedThreadPool(threads)
        val workers = Array(threads) { model.newWorker() }
        val master = model.newWorker() // for sampling / grad accumulation
        val rng = Random(args.seed + 1)

        val maxStart = stream.size - args.seqLen - 1
        val batchSeqs = Array(args.batch) { IntArray(args.seqLen + 1) }

        println("Training for ${args.steps} steps, batch ${args.batch} x ${args.seqLen} tokens...")
        val t0 = System.currentTimeMillis()
        var emaLoss = 0f
        try {
            for (step in 1..args.steps) {
                // sample random windows
                for (b in 0 until args.batch) {
                    val s = rng.nextInt(maxStart + 1)
                    stream.copyInto(batchSeqs[b], 0, s, s + args.seqLen + 1)
                }
                // parallel forward+backward
                val futures = (0 until threads).map { t ->
                    pool.submit(Callable {
                        val w = workers[t]
                        w.zeroGrads()
                        var lossSum = 0f
                        var count = 0
                        var b = t
                        while (b < args.batch) {
                            lossSum += w.forward(batchSeqs[b])
                            w.backward(batchSeqs[b])
                            count++
                            b += threads
                        }
                        doubleArrayOf(lossSum.toDouble(), count.toDouble())
                    })
                }
                var lossSum = 0.0
                var count = 0
                val parts = futures.map { it.get() }
                for ((li, wi) in parts.withIndex()) {
                    lossSum += wi[0]
                    count += wi[1].toInt()
                    // accumulate worker grads into master
                    val wg = workers[li].grads
                    val mg = master.grads
                    for (i in mg.indices) addInto(mg[i], wg[i])
                }
                // mean over batch
                for (g in master.grads) scaleInto(g, 1f / args.batch)
                val stepLoss = (lossSum / count).toFloat()
                emaLoss = if (step == 1) stepLoss else 0.98f * emaLoss + 0.02f * stepLoss

                val gNorm = adam.gradNorm(master.grads)
                adam.step(master.grads)
                master.zeroGrads()

                if (step % args.logEvery == 0 || step == 1) {
                    val el = (System.currentTimeMillis() - t0) / 1000.0
                    println("step %5d/%d  loss %.4f (ema %.4f)  %.1fs  gradNorm %.3f".format(
                        step, args.steps, stepLoss, emaLoss, el, gNorm
                    ))
                }
                if (step % args.sampleEvery == 0) {
                    val sample = Generate.sample(
                        model, tok, master,
                        prompt = "the king said",
                        maxTokens = 40, temperature = 0.9f, topK = 40,
                        rng = Random(args.seed + step)
                    )
                    println("--- sample @ step $step ---\n$sample\n----------------------")
                }
                if (args.checkpointEvery > 0 && step % args.checkpointEvery == 0) {
                    model.save(File(args.out + ".ckpt-$step"))
                    println("checkpoint saved: ${args.out}.ckpt-$step")
                }
            }
        } finally {
            pool.shutdown()
        }
        val secs = (System.currentTimeMillis() - t0) / 1000.0
        println("Done: ${args.steps} steps in %.1f s (%.2f steps/s)".format(secs, args.steps / secs))
        println("Final EMA loss: %.4f (random would be ~%.2f)".format(
            emaLoss, kotlin.math.ln(tok.vocabSize.toDouble())))
        model.save(File(args.out))
        tok.save(File(args.vocabOut))
        println("Saved weights -> ${args.out}, vocab -> ${args.vocabOut}")
    }
}
