package forgegpt

/**
 * ForgeGPT: a from-scratch decoder-only transformer language model.
 * Pure Kotlin, zero ML frameworks — every matrix multiply, softmax,
 * layer norm, attention head and Adam update is hand-written.
 *
 * Usage:
 *   java -jar forgegpt.jar train    [options]   - train on the corpus
 *   java -jar forgegpt.jar generate [options]   - generate text
 *   java -jar forgegpt.jar gradcheck             - verify backprop math
 *   java -jar forgegpt.jar gui                   - launch the Swing GUI
 */
fun main(argv: Array<String>) {
    if (argv.isEmpty()) {
        printUsage()
        return
    }
    when (argv[0].lowercase()) {
        "train" -> Train.run(Train.parse(argv.drop(1)))
        "generate" -> Generate.run(Generate.parse(argv.drop(1)))
        "gradcheck" -> {
            if (!GradCheck.run()) kotlin.system.exitProcess(1)
        }
        "dumpgrad" -> GradCheck.dump(argv.getOrElse(1) { "/tmp/gc_dump.txt" })
        "gui" -> forgegpt.gui.launchGui()
        else -> {
            println("unknown command: ${argv[0]}")
            printUsage()
        }
    }
}

private fun printUsage() {
    println(
        """
        ForgeGPT - tiny from-scratch transformer language model.
        
        Commands:
          train     Train a model on a text corpus.
                    --corpus FILE        training text (default corpus/corpus.txt)
                    --out FILE           weights output (default model.fg)
                    --vocab-out FILE     vocab output (default vocab.txt)
                    --steps N            training steps (default 600)
                    --batch N            sequences per step (default 8)
                    --lr F               Adam learning rate (default 3e-4)
                    --d-model N          embedding width (default 64)
                    --layers N           transformer blocks (default 3)
                    --heads N            attention heads (default 4)
                    --seq-len N          context length (default 48)
                    --ffn N              MLP hidden size (default 256)
                    --vocab N            max vocabulary size (default 2500)
                    --seed N             random seed (default 42)
                    --no-gradcheck       skip the backprop self-test
                    --init-model FILE    resume training from a saved model.fg
        
          generate  Generate text from a trained model.
                    --model FILE         weights (default model.fg)
                    --vocab FILE         vocab (default vocab.txt)
                    --prompt "..."       prompt text
                    --max-tokens N       tokens to generate (default 80)
                    --temperature F      sampling temperature (default 0.8)
                    --top-k N            top-k sampling (default 40)
                    --seed N             sampling seed (default 7)
        
          gradcheck Verify the hand-written backprop with finite differences.

          gui       Launch the Swing desktop GUI (Generate / Train / Corpus /
                    Model Lab / Logs tabs over the same engine).
        """.trimIndent()
    )
}
