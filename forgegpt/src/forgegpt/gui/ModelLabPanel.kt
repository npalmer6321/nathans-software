package forgegpt.gui

import forgegpt.Engine
import forgegpt.EvalResult
import forgegpt.GPT
import forgegpt.GradCheck
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSplitPane
import javax.swing.JTable
import javax.swing.JTextArea
import javax.swing.SwingWorker
import javax.swing.table.DefaultTableModel

/**
 * Model Lab tab: inspect the loaded model (config, per-component
 * parameter counts), run a validation/perplexity mini-evaluation on
 * pasted text, and run the numerical gradient check.
 *
 * Evaluation and gradcheck run on background threads; gradcheck output
 * is captured by temporarily redirecting stdout, since [GradCheck]
 * reports via println.
 */
class ModelLabPanel(private val state: AppState) : JPanel(BorderLayout()) {

    private val infoModel = DefaultTableModel(arrayOf("Property", "Value"), 0)
    private val infoTable = JTable(infoModel)
    private val breakdownModel = DefaultTableModel(arrayOf("Component", "Parameters", "Share"), 0)
    private val breakdownTable = JTable(breakdownModel)
    private val infoStatus = JLabel("No model loaded.")

    private val evalInput = JTextArea(5, 50).apply {
        font = Ui.MONO
        lineWrap = true
        wrapStyleWord = true
        text = "Alice was beginning to get very tired of sitting by her sister on the bank, " +
            "and of having nothing to do: once or twice she had peeped into the book her sister was reading, " +
            "but it had no pictures or conversations in it, \"and what is the use of a book,\" thought Alice " +
            "\"without pictures or conversations?\" So she was considering in her own mind (as well as she could, " +
            "for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain " +
            "would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with " +
            "pink eyes ran close by her."
    }
    private val evalOutput = JTextArea(7, 50).apply {
        font = Ui.MONO
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
    }
    private val evalButton = JButton("Run evaluation")
    private val gradButton = JButton("Run gradcheck")
    private val gradOutput = JTextArea(8, 50).apply {
        font = Ui.MONO
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
    }
    private val reloadButton = JButton("Reload model from paths")
    private val saveCopyButton = JButton("Save a copy of weights...")

    init {
        border = BorderFactory.createEmptyBorder(8, 8, 8, 8)

        val topBar = JPanel(FlowLayout(FlowLayout.LEFT, 6, 4))
        topBar.add(reloadButton)
        topBar.add(saveCopyButton)
        topBar.add(infoStatus)
        add(topBar, BorderLayout.NORTH)

        val infoSplit = JSplitPane(
            JSplitPane.HORIZONTAL_SPLIT,
            Ui.titled("Model info", JScrollPane(infoTable)),
            Ui.titled("Parameter breakdown", JScrollPane(breakdownTable))
        )
        infoSplit.resizeWeight = 0.5

        val evalPanel = JPanel(BorderLayout())
        evalPanel.border = BorderFactory.createTitledBorder("Validation / perplexity mini-eval")
        evalPanel.add(JScrollPane(evalInput), BorderLayout.NORTH)
        val evalControls = JPanel(FlowLayout(FlowLayout.LEFT, 6, 2))
        evalControls.add(evalButton)
        evalControls.add(JLabel("Scores pasted text with the loaded model (lower perplexity = more predictable to it)."))
        evalPanel.add(evalControls, BorderLayout.CENTER)
        evalPanel.add(JScrollPane(evalOutput), BorderLayout.SOUTH)

        val gradPanel = JPanel(BorderLayout())
        gradPanel.border = BorderFactory.createTitledBorder("Backprop gradient check (finite differences)")
        val gradControls = JPanel(FlowLayout(FlowLayout.LEFT, 6, 2))
        gradControls.add(gradButton)
        gradControls.add(JLabel("Verifies the hand-written backprop on 0/1/2-block toy configs."))
        gradPanel.add(gradControls, BorderLayout.NORTH)
        gradPanel.add(JScrollPane(gradOutput), BorderLayout.CENTER)

        val center = JSplitPane(JSplitPane.VERTICAL_SPLIT, infoSplit, evalPanel)
        center.resizeWeight = 0.5
        add(center, BorderLayout.CENTER)
        add(gradPanel, BorderLayout.SOUTH)

        reloadButton.addActionListener { reloadModel() }
        saveCopyButton.addActionListener { saveWeightsCopy() }
        evalButton.addActionListener { runEval() }
        gradButton.addActionListener { runGradCheck() }
    }

    /** Refresh all model displays after a load elsewhere in the app. */
    fun onModelChanged() {
        refreshInfo()
    }

    private fun reloadModel() {
        try {
            val loaded = state.loadModelFromPaths()
            refreshInfo()
            infoStatus.text = "Loaded: ${loaded.modelFile.name} (${Ui.fmtBytes(loaded.modelFile.length())})"
        } catch (e: Exception) {
            infoStatus.text = "Load failed: ${e.message}"
            state.log("lab", "ERROR loading model: ${e.message}")
        }
    }

    private fun refreshInfo() {
        infoModel.rowCount = 0
        breakdownModel.rowCount = 0
        val loaded = state.requireLoaded()
        if (loaded == null) {
            infoStatus.text = "No model loaded."
            return
        }
        val c = loaded.config
        val modelFile = loaded.modelFile
        val vocabFile = loaded.vocabFile
        fun row(k: String, v: String) = infoModel.addRow(arrayOf<Any>(k, v))
        row("Model file", modelFile.path)
        row("Model file size", Ui.fmtBytes(modelFile.length()))
        row("Vocab file", vocabFile.path)
        row("Vocab size", Ui.fmtInt(c.vocab))
        row("dModel (embedding width)", Ui.fmtInt(c.dModel))
        row("Layers", Ui.fmtInt(c.nLayers))
        row("Heads", Ui.fmtInt(c.nHeads))
        row("Head dim", Ui.fmtInt(c.headDim))
        row("Context (seqLen)", Ui.fmtInt(c.seqLen) + " tokens")
        row("FFN hidden", Ui.fmtInt(c.ffnHidden))
        row("Total parameters", Ui.fmtInt(loaded.numParams))
        row("Est. size at float32", Ui.fmtBytes(Engine.estimatedModelBytes(loaded.model)))
        row("Random-guess loss for vocab", Ui.fmtFloat(Engine.randomLoss(c.vocab), 3))

        val bd = Engine.paramBreakdown(loaded.model)
        val total = bd.totalParams.coerceAtLeast(1)
        for ((label, count) in bd.rows()) {
            val share = if (label == "TOTAL") "" else Ui.fmtFloat(count * 100.0 / total, 2) + " %"
            breakdownModel.addRow(arrayOf<Any>(label, Ui.fmtInt(count), share))
        }
        infoStatus.text = "Loaded: ${modelFile.name} — ${loaded.describe()}"
    }

    private fun saveWeightsCopy() {
        val loaded = state.requireLoaded()
        if (loaded == null) {
            infoStatus.text = "No model loaded to save."
            return
        }
        val file = Ui.chooseFile(this, "Save weights copy", listOf("fg"), save = true) ?: return
        val target = if (file.name.endsWith(".fg")) file else File(file.path + ".fg")
        loaded.model.save(target)
        infoStatus.text = "Saved weights copy to ${target.path}"
        state.log("lab", "Saved weights copy: ${target.path} (${Ui.fmtBytes(target.length())})")
    }

    private fun runEval() {
        val loaded = state.requireLoaded()
        if (loaded == null) {
            evalOutput.text = "Load a model first."
            return
        }
        val text = evalInput.text
        evalButton.isEnabled = false
        evalOutput.text = "Scoring..."
        val worker = loaded.newWorker()
        val appState = state
        val task = object : SwingWorker<EvalResult, Unit>() {
            override fun doInBackground(): EvalResult = Engine.evaluate(loaded, worker, text)

            override fun done() {
                evalButton.isEnabled = true
                try {
                    val r = get()
                    evalOutput.text = buildString {
                        append("windows scored: ${r.windows} (of ${loaded.config.seqLen} tokens each)\n")
                        append("mean loss: ${Ui.fmtFloat(r.meanLoss)}   (random guess for this vocab: " +
                            "${Ui.fmtFloat(Engine.randomLoss(loaded.config.vocab), 3)})\n")
                        append("perplexity: ${Ui.fmtFloat(r.perplexity, 2)}\n")
                        append("best window loss: ${Ui.fmtFloat(r.bestWindowLoss)}   worst: ${Ui.fmtFloat(r.worstWindowLoss)}\n")
                        append("tokens scored: ${Ui.fmtInt(r.tokensScored)}   in ${Ui.fmtDuration(r.elapsedMs)}\n\n")
                        append("Reading guide: perplexity near the vocab size (${loaded.config.vocab}) means the model is guessing. ")
                        append("The demo model scores in the tens on in-corpus text and much worse on text unlike its books.")
                    }
                    appState.log("lab", "Eval: loss ${Ui.fmtFloat(r.meanLoss)}, ppl ${Ui.fmtFloat(r.perplexity, 2)} over ${r.windows} windows.")
                } catch (e: Exception) {
                    val cause = e.cause ?: e
                    evalOutput.text = "Evaluation failed: ${cause.message}"
                    appState.log("lab", "ERROR eval: ${cause.message}")
                }
            }
        }
        task.execute()
    }

    private fun runGradCheck() {
        gradButton.isEnabled = false
        gradOutput.text = "Running numerical gradient check (builds toy models, perturbs every parameter)...\n"
        val appState = state
        val task = object : SwingWorker<Pair<Boolean, String>, Unit>() {
            override fun doInBackground(): Pair<Boolean, String> {
                val buffer = ByteArrayOutputStream()
                val capture = PrintStream(buffer, true, "UTF-8")
                val original = System.out
                var ok = false
                try {
                    System.setOut(capture)
                    ok = GradCheck.run()
                } finally {
                    System.setOut(original)
                    capture.flush()
                }
                return ok to buffer.toString("UTF-8")
            }

            override fun done() {
                gradButton.isEnabled = true
                try {
                    val (ok, text) = get()
                    gradOutput.text = text + if (ok) "\nAll configs passed — the backprop math checks out." else "\nFAILED — see relative errors above."
                    gradOutput.caretPosition = 0
                    appState.log("lab", "Gradcheck: ${if (ok) "PASS" else "FAIL"}")
                } catch (e: Exception) {
                    val cause = e.cause ?: e
                    gradOutput.text = "Gradcheck crashed: ${cause.message}"
                    appState.log("lab", "ERROR gradcheck: ${cause.message}")
                }
            }
        }
        task.execute()
    }
}
