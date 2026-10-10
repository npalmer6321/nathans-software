package forgegpt

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.random.Random

/**
 * Trainer — the callback-driven training engine used by the GUI's Train tab.
 *
 * [Train] (the CLI path) keeps its own print-to-stdout loop; this class
 * runs the identical algorithm (random windows, parallel forward/backward
 * across worker threads, mean gradient, one Adam step per batch) but
 * reports through [TrainListener] so a UI can show live progress, a loss
 * readout, periodic samples, and a working Stop button.
 */
class Trainer {

    /** Everything the training loop needs, mirroring [Train.Args]. */
    data class Settings(
        val corpusPath: String = "corpus/corpus.txt",
        val outPath: String = "model.fg",
        val vocabOutPath: String = "vocab.txt",
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
        val threads: Int = 0, // 0 = auto (1..4 based on CPU count)
        val initModelPath: String = "",
        val runGradCheckFirst: Boolean = true,
        val samplePrompt: String = "the king said"
    ) {
        fun toConfig(vocabSize: Int): GPTConfig = GPTConfig(
            vocab = vocabSize, dModel = dModel, nLayers = nLayers,
            nHeads = nHeads, seqLen = seqLen, ffnHidden = ffnHidden, seed = seed
        )

        companion object {
            /** Sensible defaults matching the shipped demo model's shape. */
            fun demoDefaults(): Settings = Settings()

            /** A quick smoke-run preset: tiny and fast, for verifying a setup. */
            fun smokeDefaults(): Settings = Settings(
                steps = 30, batch = 4, dModel = 32, nLayers = 2, nHeads = 4,
                seqLen = 32, ffnHidden = 128, logEvery = 5, sampleEvery = 15
            )
        }
    }

    /** Callbacks fired from the training thread. Implementations must be thread-safe. */
    interface TrainListener {
        fun onLog(message: String)
        fun onStep(step: Int, totalSteps: Int, loss: Float, emaLoss: Float, gradNorm: Float, elapsedSec: Double)
        fun onSample(step: Int, text: String)
        fun onCheckpoint(step: Int, path: String)
        fun onFinished(savedModel: File?, savedVocab: File?, message: String)
    }

    /** Snapshot of a finished (or stopped) run, returned from [run]. */
    data class Outcome(
        val completed: Boolean,
        val stepsRun: Int,
        val finalEmaLoss: Float,
        val seconds: Double,
        val model: GPT?,
        val tokenizer: Tokenizer?
    )

    @Volatile
    var stopRequested: Boolean = false
        private set

    fun requestStop() {
        stopRequested = true
    }

    /**
     * Run training to completion (or until [requestStop]). Blocks the
     * calling thread — the GUI runs this on a background worker.
     */
    fun run(settings: Settings, listener: TrainListener): Outcome {
        stopRequested = false
        fun log(msg: String) = listener.onLog(msg)

        if (settings.runGradCheckFirst) {
            log("Running gradient check first (verifies the hand-written backprop)...")
            if (!GradCheck.run()) {
                val msg = "Gradient check FAILED - refusing to train on broken math."
                log(msg)
                listener.onFinished(null, null, msg)
                return Outcome(false, 0, Float.NaN, 0.0, null, null)
            }
        }

        val corpusFile = File(settings.corpusPath)
        if (!corpusFile.isFile) {
            val msg = "Corpus not found: ${settings.corpusPath}"
            log(msg)
            listener.onFinished(null, null, msg)
            return Outcome(false, 0, Float.NaN, 0.0, null, null)
        }
        log("Loading corpus: ${settings.corpusPath}")
        val text = corpusFile.readText()
        log("Building vocabulary (max ${settings.maxVocab})...")
        val tok = Tokenizer.buildFromCorpus(text, settings.maxVocab)
        log("Vocab size: ${tok.vocabSize}")
        log("Tokenizing corpus...")
        val stream = Tokenizer.tokenizeText(text)
            .map { tok.tokenToId[it] ?: Tokenizer.UNK }.toIntArray()
        log("Corpus tokens: ${stream.size}")
        if (stream.size <= settings.seqLen + 1) {
            val msg = "Corpus too small (${stream.size} tokens for seqLen ${settings.seqLen})."
            log(msg)
            listener.onFinished(null, null, msg)
            return Outcome(false, 0, Float.NaN, 0.0, null, null)
        }

        val cfg = settings.toConfig(tok.vocabSize)
        val model: GPT
        if (settings.initModelPath.isNotEmpty()) {
            log("Resuming from weights: ${settings.initModelPath}")
            val m = GPT.load(File(settings.initModelPath))
            val c = m.config
            val matches = c.vocab == cfg.vocab && c.dModel == cfg.dModel && c.nLayers == cfg.nLayers &&
                c.seqLen == cfg.seqLen && c.ffnHidden == cfg.ffnHidden && c.nHeads == cfg.nHeads
            if (!matches) {
                val msg = "init-model config does not match training config " +
                    "(model: ${c.vocab}/${c.dModel}/${c.nLayers}L, requested: ${cfg.vocab}/${cfg.dModel}/${cfg.nLayers}L)"
                log(msg)
                listener.onFinished(null, null, msg)
                return Outcome(false, 0, Float.NaN, 0.0, null, null)
            }
            model = m
        } else {
            model = GPT(cfg)
        }
        log("Parameters: ${"%,d".format(model.numParams)}")
        log("Config: ${loadedDescribe(cfg)}")
        val adam = Adam(model, lr = settings.lr)

        val threads = if (settings.threads > 0) settings.threads
        else maxOf(1, minOf(4, Runtime.getRuntime().availableProcessors()))
        log("Worker threads: $threads")
        val pool = Executors.newFixedThreadPool(threads)
        val workers = Array(threads) { model.newWorker() }
        val master = model.newWorker() // grad accumulation + sampling
        val rng = Random(settings.seed + 1)

        val maxStart = stream.size - settings.seqLen - 1
        val batchSeqs = Array(settings.batch) { IntArray(settings.seqLen + 1) }

        log("Training for ${settings.steps} steps, batch ${settings.batch} x ${settings.seqLen} tokens...")
        val t0 = System.currentTimeMillis()
        var emaLoss = 0f
        var stepsRun = 0
        try {
            for (step in 1..settings.steps) {
                if (stopRequested) {
                    log("Stop requested at step ${step - 1}; finishing early.")
                    break
                }
                stepsRun = step
                for (b in 0 until settings.batch) {
                    val s = rng.nextInt(maxStart + 1)
                    stream.copyInto(batchSeqs[b], 0, s, s + settings.seqLen + 1)
                }
                val futures = (0 until threads).map { t ->
                    pool.submit(Callable {
                        val w = workers[t]
                        w.zeroGrads()
                        var lossSum = 0f
                        var count = 0
                        var b = t
                        while (b < settings.batch) {
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
                    val wg = workers[li].grads
                    val mg = master.grads
                    for (i in mg.indices) addInto(mg[i], wg[i])
                }
                for (g in master.grads) scaleInto(g, 1f / settings.batch)
                val stepLoss = (lossSum / count).toFloat()
                emaLoss = if (step == 1) stepLoss else 0.98f * emaLoss + 0.02f * stepLoss

                val gNorm = adam.gradNorm(master.grads)
                adam.step(master.grads)
                master.zeroGrads()

                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                if (step % settings.logEvery == 0 || step == 1) {
                    log(
                        "step %5d/%d  loss %.4f (ema %.4f)  %.1fs  gradNorm %.3f".format(
                            step, settings.steps, stepLoss, emaLoss, elapsed, gNorm
                        )
                    )
                }
                listener.onStep(step, settings.steps, stepLoss, emaLoss, gNorm, elapsed)

                if (settings.sampleEvery > 0 && step % settings.sampleEvery == 0) {
                    val sample = Generate.sample(
                        model, tok, master,
                        prompt = settings.samplePrompt,
                        maxTokens = 40, temperature = 0.9f, topK = 40,
                        rng = Random(settings.seed + step)
                    )
                    listener.onSample(step, sample)
                }
                if (settings.checkpointEvery > 0 && step % settings.checkpointEvery == 0) {
                    val ckpt = File(settings.outPath + ".ckpt-$step")
                    model.save(ckpt)
                    log("checkpoint saved: ${ckpt.path}")
                    listener.onCheckpoint(step, ckpt.path)
                }
            }
        } finally {
            pool.shutdown()
        }
        val secs = (System.currentTimeMillis() - t0) / 1000.0
        val outFile = File(settings.outPath)
        val vocabFile = File(settings.vocabOutPath)
        model.save(outFile)
        tok.save(vocabFile)
        val msg = "Done: $stepsRun steps in %.1f s (%.2f steps/s). Final EMA loss %.4f (random ~%.2f). Saved -> ${outFile.path}, vocab -> ${vocabFile.path}"
            .format(secs, if (secs > 0) stepsRun / secs else 0.0, emaLoss, Engine.randomLoss(tok.vocabSize))
        log(msg)
        listener.onFinished(outFile, vocabFile, msg)
        return Outcome(true, stepsRun, emaLoss, secs, model, tok)
    }

    private fun loadedDescribe(c: GPTConfig): String =
        "vocab=${c.vocab}, dModel=${c.dModel}, layers=${c.nLayers}, heads=${c.nHeads}, seqLen=${c.seqLen}, ffn=${c.ffnHidden}, seed=${c.seed}"
}
