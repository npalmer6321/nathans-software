package forgegpt.gui

import forgegpt.Trainer
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Graphics
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.io.File
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JCheckBox
import javax.swing.JComponent
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JProgressBar
import javax.swing.JScrollPane
import javax.swing.JSpinner
import javax.swing.JTextField
import javax.swing.SpinnerNumberModel
import javax.swing.SwingUtilities
import javax.swing.SwingWorker

/**
 * Train tab: full hyperparameter form, live progress (step bar, loss,
 * EMA loss, grad norm, steps/sec), a small loss-curve chart, periodic
 * sample generations, and checkpoint save/load.
 *
 * Training runs on a background [SwingWorker] driving [Trainer]; all UI
 * updates come back through the Trainer listener on the EDT.
 */
class TrainPanel(private val state: AppState) : JPanel(BorderLayout()) {

    // ---- form fields ----
    private val corpusField = JTextField(state.corpusPath)
    private val outField = JTextField("model.fg")
    private val vocabOutField = JTextField("vocab.txt")
    private val initModelField = JTextField("")
    private val stepsSpinner = JSpinner(SpinnerNumberModel(600, 1, 1_000_000, 50))
    private val batchSpinner = JSpinner(SpinnerNumberModel(8, 1, 512, 1))
    private val lrField = JTextField("0.0003")
    private val dModelSpinner = JSpinner(SpinnerNumberModel(64, 8, 2048, 8))
    private val layersSpinner = JSpinner(SpinnerNumberModel(3, 0, 48, 1))
    private val headsSpinner = JSpinner(SpinnerNumberModel(4, 1, 64, 1))
    private val seqLenSpinner = JSpinner(SpinnerNumberModel(48, 4, 4096, 4))
    private val ffnSpinner = JSpinner(SpinnerNumberModel(256, 8, 8192, 8))
    private val vocabSpinner = JSpinner(SpinnerNumberModel(2500, 16, 200000, 100))
    private val seedField = JTextField("42")
    private val threadsSpinner = JSpinner(SpinnerNumberModel(0, 0, 64, 1))
    private val logEverySpinner = JSpinner(SpinnerNumberModel(25, 1, 100000, 5))
    private val sampleEverySpinner = JSpinner(SpinnerNumberModel(200, 0, 100000, 10))
    private val checkpointEverySpinner = JSpinner(SpinnerNumberModel(0, 0, 100000, 50))
    private val gradCheckBox = JCheckBox("Run gradcheck before training", true)
    private val samplePromptField = JTextField("the king said")

    // ---- live readouts ----
    private val progressBar = JProgressBar(0, 100).apply { isStringPainted = true }
    private val lossLabel = JLabel("loss: -")
    private val emaLabel = JLabel("ema: -")
    private val gradLabel = JLabel("gradNorm: -")
    private val rateLabel = JLabel("rate: -")
    private val etaLabel = JLabel("eta: -")

    private val chart = LossChart()
    private val sampleArea = javax.swing.JTextArea(6, 60).apply {
        font = Ui.MONO
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
    }
    private val logView = LogView("Training log")

    private val startButton = JButton("Start training")
    private val stopButton = JButton("Stop").apply { isEnabled = false }
    private val smokeButton = JButton("Smoke run")
    private val loadCheckpointButton = JButton("Use checkpoint as init...")

    private var trainer: Trainer? = null
    private var task: SwingWorker<Trainer.Outcome, Unit>? = null
    private var runStartedAt: Long = 0L

    init {
        border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        val top = JPanel(BorderLayout())
        top.add(buildForm(), BorderLayout.NORTH)
        top.add(buildReadouts(), BorderLayout.CENTER)
        add(top, BorderLayout.NORTH)

        val bottom = JPanel(BorderLayout())
        val chartPanel = Ui.titled("Loss curve (EMA)", chart)
        chartPanel.preferredSize = Dimension(300, 140)
        bottom.add(chartPanel, BorderLayout.WEST)
        val samplePanel = Ui.titled("Latest sample", JScrollPane(sampleArea))
        bottom.add(samplePanel, BorderLayout.CENTER)
        add(bottom, BorderLayout.CENTER)
        add(logView, BorderLayout.SOUTH)
        logView.preferredSize = Dimension(640, 130)

        startButton.addActionListener { startTraining(smoke = false) }
        smokeButton.addActionListener { startTraining(smoke = true) }
        stopButton.addActionListener {
            trainer?.requestStop()
            log("Stop requested — the current step will finish, then training wraps up and saves.")
        }
        loadCheckpointButton.addActionListener {
            val f = Ui.chooseFile(this, "Checkpoint / model to resume from", listOf("fg")) ?: return@addActionListener
            initModelField.text = f.path
            log("Init model set to ${f.path}")
        }
    }

    private fun buildForm(): JPanel {
        val form = JPanel(GridBagLayout())
        form.border = BorderFactory.createTitledBorder("Training setup")
        val c = GridBagConstraints().apply {
            insets = Insets(2, 4, 2, 4)
            anchor = GridBagConstraints.WEST
        }
        var row = 0
        fun addRow(label: String, field: JComponent, browse: (() -> Unit)? = null, weight: Double = 1.0) {
            c.gridy = row; c.gridx = 0; c.weightx = 0.0; c.fill = GridBagConstraints.NONE
            form.add(Ui.label(label), c)
            c.gridx = 1; c.weightx = weight; c.fill = GridBagConstraints.HORIZONTAL
            form.add(field, c)
            if (browse != null) {
                c.gridx = 2; c.weightx = 0.0; c.fill = GridBagConstraints.NONE
                val b = JButton("...")
                b.addActionListener { browse() }
                form.add(b, c)
            }
            row++
        }
        corpusField.columns = 42
        addRow("Corpus:", corpusField, {
            Ui.chooseFile(this, "Training corpus", listOf("txt"))?.let { corpusField.text = it.path }
        })
        addRow("Weights out:", outField, {
            Ui.chooseFile(this, "Weights output", listOf("fg"), save = true)?.let { outField.text = it.path }
        })
        addRow("Vocab out:", vocabOutField, {
            Ui.chooseFile(this, "Vocab output", listOf("txt"), save = true)?.let { vocabOutField.text = it.path }
        })
        addRow("Init model (resume):", initModelField, {
            Ui.chooseFile(this, "Init model", listOf("fg"))?.let { initModelField.text = it.path }
        })

        val hyper = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
            fun pair(label: String, comp: JComponent) {
                add(Ui.label(label)); add(comp)
            }
            pair("Steps:", stepsSpinner)
            pair("Batch:", batchSpinner)
            pair("LR:", lrField.apply { columns = 7 })
            pair("dModel:", dModelSpinner)
            pair("Layers:", layersSpinner)
            pair("Heads:", headsSpinner)
        }
        c.gridy = row++; c.gridx = 0; c.gridwidth = 3; c.weightx = 1.0; c.fill = GridBagConstraints.HORIZONTAL
        form.add(hyper, c)
        val hyper2 = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
            fun pair(label: String, comp: JComponent) {
                add(Ui.label(label)); add(comp)
            }
            pair("SeqLen:", seqLenSpinner)
            pair("FFN:", ffnSpinner)
            pair("Vocab:", vocabSpinner)
            pair("Seed:", seedField.apply { columns = 6 })
            pair("Threads (0=auto):", threadsSpinner)
        }
        c.gridy = row++; form.add(hyper2, c)
        val hyper3 = JPanel(FlowLayout(FlowLayout.LEFT, 8, 2)).apply {
            fun pair(label: String, comp: JComponent) {
                add(Ui.label(label)); add(comp)
            }
            pair("Log every:", logEverySpinner)
            pair("Sample every:", sampleEverySpinner)
            pair("Checkpoint every:", checkpointEverySpinner)
            add(gradCheckBox)
        }
        c.gridy = row++; form.add(hyper3, c)
        addRow("Sample prompt:", samplePromptField, null)

        val buttons = JPanel(FlowLayout(FlowLayout.LEFT, 6, 4)).apply {
            add(startButton)
            add(stopButton)
            add(smokeButton)
            add(loadCheckpointButton)
        }
        c.gridy = row; c.gridx = 0; c.gridwidth = 3
        form.add(buttons, c)
        return form
    }

    private fun buildReadouts(): JPanel {
        val panel = JPanel(FlowLayout(FlowLayout.LEFT, 14, 4))
        panel.border = BorderFactory.createTitledBorder("Progress")
        panel.add(progressBar.apply { preferredSize = Dimension(220, 20) })
        panel.add(lossLabel)
        panel.add(emaLabel)
        panel.add(gradLabel)
        panel.add(rateLabel)
        panel.add(etaLabel)
        return panel
    }

    private fun log(msg: String) {
        logView.append(msg)
        state.log("train", msg)
    }

    private fun readSettings(smoke: Boolean): Trainer.Settings {
        val base = if (smoke) Trainer.Settings.smokeDefaults() else Trainer.Settings()
        return base.copy(
            corpusPath = corpusField.text.trim().ifEmpty { state.corpusPath },
            outPath = outField.text.trim().ifEmpty { "model.fg" },
            vocabOutPath = vocabOutField.text.trim().ifEmpty { "vocab.txt" },
            initModelPath = initModelField.text.trim(),
            steps = if (smoke) base.steps else (stepsSpinner.value as Number).toInt(),
            batch = if (smoke) base.batch else (batchSpinner.value as Number).toInt(),
            lr = lrField.text.trim().toFloatOrNull() ?: 3e-4f,
            dModel = if (smoke) base.dModel else (dModelSpinner.value as Number).toInt(),
            nLayers = if (smoke) base.nLayers else (layersSpinner.value as Number).toInt(),
            nHeads = if (smoke) base.nHeads else (headsSpinner.value as Number).toInt(),
            seqLen = if (smoke) base.seqLen else (seqLenSpinner.value as Number).toInt(),
            ffnHidden = if (smoke) base.ffnHidden else (ffnSpinner.value as Number).toInt(),
            maxVocab = (vocabSpinner.value as Number).toInt(),
            seed = seedField.text.trim().toLongOrNull() ?: 42L,
            threads = (threadsSpinner.value as Number).toInt(),
            logEvery = (logEverySpinner.value as Number).toInt(),
            sampleEvery = (sampleEverySpinner.value as Number).toInt(),
            checkpointEvery = (checkpointEverySpinner.value as Number).toInt(),
            runGradCheckFirst = gradCheckBox.isSelected,
            samplePrompt = samplePromptField.text.ifBlank { "the king said" }
        )
    }

    private fun validate(settings: Trainer.Settings): String? {
        if (!File(settings.corpusPath).isFile) return "Corpus file not found: ${settings.corpusPath}"
        if (settings.dModel % settings.nHeads != 0) {
            return "dModel (${settings.dModel}) must be divisible by heads (${settings.nHeads})."
        }
        if (settings.initModelPath.isNotEmpty() && !File(settings.initModelPath).isFile) {
            return "Init model not found: ${settings.initModelPath}"
        }
        return null
    }

    private fun startTraining(smoke: Boolean) {
        if (task != null && !task!!.isDone) {
            log("Training already running; ignoring duplicate start.")
            return
        }
        val settings = readSettings(smoke)
        val problem = validate(settings)
        if (problem != null) {
            JOptionPane.showMessageDialog(this, problem, "Can't start training", JOptionPane.WARNING_MESSAGE)
            log("Start refused: $problem")
            return
        }
        state.corpusPath = settings.corpusPath

        val newTrainer = Trainer()
        trainer = newTrainer
        chart.clear()
        sampleArea.text = ""
        progressBar.value = 0
        progressBar.string = "starting..."
        lossLabel.text = "loss: -"
        emaLabel.text = "ema: -"
        gradLabel.text = "gradNorm: -"
        rateLabel.text = "rate: -"
        etaLabel.text = "eta: -"
        runStartedAt = System.currentTimeMillis()
        startButton.isEnabled = false
        smokeButton.isEnabled = false
        stopButton.isEnabled = true

        val listener = object : Trainer.TrainListener {
            override fun onLog(message: String) {
                SwingUtilities.invokeLater { log(message) }
            }

            override fun onStep(
                step: Int, totalSteps: Int, loss: Float, emaLoss: Float,
                gradNorm: Float, elapsedSec: Double
            ) {
                SwingUtilities.invokeLater {
                    val pct = if (totalSteps > 0) (100.0 * step / totalSteps).toInt() else 0
                    progressBar.maximum = 100
                    progressBar.value = pct
                    progressBar.string = "step $step / $totalSteps ($pct%)"
                    lossLabel.text = "loss: ${Ui.fmtFloat(loss)}"
                    emaLabel.text = "ema: ${Ui.fmtFloat(emaLoss)}"
                    gradLabel.text = "gradNorm: ${Ui.fmtFloat(gradNorm, 3)}"
                    val rate = if (elapsedSec > 0) step / elapsedSec else 0.0
                    rateLabel.text = "rate: ${Ui.fmtFloat(rate, 2)} steps/s"
                    val remainSteps = totalSteps - step
                    val etaSec = if (rate > 0) remainSteps / rate else 0.0
                    etaLabel.text = "eta: ${Ui.fmtDuration((etaSec * 1000).toLong())}"
                    chart.addPoint(emaLoss.toDouble())
                }
            }

            override fun onSample(step: Int, text: String) {
                SwingUtilities.invokeLater {
                    sampleArea.text = "--- sample @ step $step ---\n$text"
                    sampleArea.caretPosition = 0
                }
            }

            override fun onCheckpoint(step: Int, path: String) {
                SwingUtilities.invokeLater { log("Checkpoint @ step $step: $path") }
            }

            override fun onFinished(savedModel: File?, savedVocab: File?, message: String) {
                SwingUtilities.invokeLater {
                    log(message)
                    if (savedModel != null && savedModel.isFile) {
                        state.modelPath = savedModel.path
                        state.vocabPath = savedVocab?.path ?: state.vocabPath
                        log("Tip: load the fresh model from the top bar or the Model Lab tab to generate with it.")
                    }
                }
            }
        }

        val worker = object : SwingWorker<Trainer.Outcome, Unit>() {
            override fun doInBackground(): Trainer.Outcome = newTrainer.run(settings, listener)

            override fun done() {
                startButton.isEnabled = true
                smokeButton.isEnabled = true
                stopButton.isEnabled = false
                try {
                    val outcome = get()
                    progressBar.string = if (outcome.completed) {
                        "done — ${outcome.stepsRun} steps in ${Ui.fmtDuration((outcome.seconds * 1000).toLong())}"
                    } else {
                        "stopped / failed after ${outcome.stepsRun} steps"
                    }
                    if (outcome.completed) progressBar.value = 100
                } catch (e: Exception) {
                    val cause = e.cause ?: e
                    log("Training crashed: ${cause.message}")
                    progressBar.string = "crashed"
                }
            }
        }
        task = worker
        log(if (smoke) "Starting SMOKE run (tiny preset)..." else "Starting training...")
        worker.execute()
    }
}

/** Minimal scrolling line chart of EMA loss. */
class LossChart : JComponent() {
    private val points = ArrayList<Double>()
    private var minV = Double.MAX_VALUE
    private var maxV = Double.MIN_VALUE

    init {
        preferredSize = Dimension(280, 140)
        toolTipText = "Exponential moving average of training loss"
    }

    fun addPoint(v: Double) {
        if (v.isNaN() || v.isInfinite()) return
        points.add(v)
        if (v < minV) minV = v
        if (v > maxV) maxV = v
        if (points.size > 5000) points.removeAt(0)
        repaint()
    }

    fun clear() {
        points.clear()
        minV = Double.MAX_VALUE
        maxV = Double.MIN_VALUE
        repaint()
    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val w = width
        val h = height
        g.color = Color(24, 26, 32)
        g.fillRect(0, 0, w, h)
        g.color = Color(90, 96, 110)
        g.drawRect(0, 0, w - 1, h - 1)
        if (points.size < 2) {
            g.color = Color(160, 166, 178)
            g.drawString("loss curve appears once training logs steps", 10, h / 2)
            return
        }
        val span = (maxV - minV).coerceAtLeast(1e-6)
        val padTop = 10.0
        val padBottom = 18.0
        val usable = h - padTop - padBottom
        g.color = Color(64, 200, 140)
        var prevX = 0
        var prevY = 0
        for (i in points.indices) {
            val x = (i.toDouble() / (points.size - 1) * (w - 2)).toInt()
            val yn = (points[i] - minV) / span
            val y = (padTop + (1.0 - yn) * usable).toInt()
            if (i > 0) g.drawLine(prevX, prevY, x, y)
            prevX = x
            prevY = y
        }
        g.color = Color(200, 205, 215)
        g.drawString("max ${Ui.fmtFloat(maxV, 2)}", 6, 14)
        g.drawString("min ${Ui.fmtFloat(minV, 2)}", 6, h - 6)
    }
}
