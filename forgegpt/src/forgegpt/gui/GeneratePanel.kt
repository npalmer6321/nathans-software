package forgegpt.gui

import forgegpt.Engine
import forgegpt.GPT
import forgegpt.GenerationResult
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.io.File
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JOptionPane
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSlider
import javax.swing.JSpinner
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.SpinnerNumberModel
import javax.swing.SwingWorker

/**
 * Generate tab: prompt in, streamed text out.
 *
 * Generation runs on a [SwingWorker] background thread using a dedicated
 * [GPT.Worker]; tokens are published back to the EDT as they are sampled
 * so the output appears to type itself. Temperature and top-k come from
 * sliders/spinners; a seed field makes runs reproducible.
 */
class GeneratePanel(private val state: AppState) : JPanel(BorderLayout()) {

    private val promptField = JTextField("alice looked")
    private val temperatureSlider = JSlider(0, 200, 80) // 0.00 .. 2.00 in hundredths
    private val temperatureLabel = JLabel("0.80")
    private val topKSpinner = JSpinner(SpinnerNumberModel(40, 1, 2500, 1))
    private val maxTokensSpinner = JSpinner(SpinnerNumberModel(120, 1, 4000, 10))
    private val seedField = JTextField(state.defaultSeed.toString())
    private val outputArea = JTextArea().apply {
        font = Ui.MONO
        lineWrap = state.wrapOutput
        wrapStyleWord = true
        isEditable = false
    }
    private val statusLabel = JLabel("No model loaded yet — load one above or use the Model Lab tab.")
    private val generateButton = JButton("Generate")
    private val stopButton = JButton("Stop").apply { isEnabled = false }
    private val copyButton = JButton("Copy")
    private val saveButton = JButton("Save...")
    private val clearButton = JButton("Clear")

    @Volatile
    private var stopRequested = false

    private var currentTask: SwingWorker<GenerationResult, String>? = null

    init {
        border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        add(buildControls(), BorderLayout.NORTH)
        add(JScrollPane(outputArea), BorderLayout.CENTER)
        add(buildFooter(), BorderLayout.SOUTH)

        temperatureSlider.addChangeListener {
            temperatureLabel.text = Ui.fmtFloat(temperatureSlider.value / 100.0, 2)
        }
        generateButton.addActionListener { startGeneration() }
        stopButton.addActionListener {
            stopRequested = true
            state.log("generate", "Stop requested by user.")
        }
        copyButton.addActionListener {
            val sel = java.awt.Toolkit.getDefaultToolkit().systemClipboard
            sel.setContents(java.awt.datatransfer.StringSelection(outputArea.text), null)
            state.log("generate", "Copied ${outputArea.text.length} chars to clipboard.")
        }
        saveButton.addActionListener { saveOutput() }
        clearButton.addActionListener { outputArea.text = "" }
    }

    private fun buildControls(): JPanel {
        val panel = JPanel(GridBagLayout())
        panel.border = BorderFactory.createTitledBorder("Generation")
        val c = GridBagConstraints().apply {
            insets = Insets(3, 4, 3, 4)
            anchor = GridBagConstraints.WEST
        }
        fun add(row: Int, col: Int, comp: java.awt.Component, weightx: Double = 0.0) {
            c.gridy = row; c.gridx = col; c.weightx = weightx
            c.fill = if (weightx > 0) GridBagConstraints.HORIZONTAL else GridBagConstraints.NONE
            panel.add(comp, c)
        }
        add(0, 0, Ui.label("Prompt:"))
        add(0, 1, promptField, 1.0)
        add(1, 0, Ui.label("Temperature:"))
        val tempPanel = JPanel(FlowLayout(FlowLayout.LEFT, 4, 0)).apply {
            add(temperatureSlider)
            add(temperatureLabel)
        }
        add(1, 1, tempPanel, 1.0)
        add(2, 0, Ui.label("Top-k:"))
        val right = JPanel(FlowLayout(FlowLayout.LEFT, 10, 0)).apply {
            add(topKSpinner)
            add(Ui.label("Max tokens:"))
            add(maxTokensSpinner)
            add(Ui.label("Seed:"))
            seedField.columns = 8
            add(seedField)
        }
        add(2, 1, right, 1.0)
        return panel
    }

    private fun buildFooter(): JPanel {
        val panel = JPanel(BorderLayout())
        val buttons = JPanel(FlowLayout(FlowLayout.LEFT, 6, 4)).apply {
            add(generateButton)
            add(stopButton)
            add(copyButton)
            add(saveButton)
            add(clearButton)
        }
        panel.add(buttons, BorderLayout.WEST)
        panel.add(statusLabel, BorderLayout.CENTER)
        return panel
    }

    /** Called when a model is (re)loaded elsewhere; refreshes the status line. */
    fun onModelChanged() {
        val loaded = state.requireLoaded()
        statusLabel.text = if (loaded == null) {
            "No model loaded yet — load one above or use the Model Lab tab."
        } else {
            "Model ready: ${loaded.describe()}"
        }
        state.log("generate", statusLabel.text)
    }

    private fun startGeneration() {
        val loaded = state.requireLoaded()
        if (loaded == null) {
            JOptionPane.showMessageDialog(
                this,
                "Load a model first (top of the window: model + vocab paths, then Load).",
                "No model loaded",
                JOptionPane.INFORMATION_MESSAGE
            )
            return
        }
        if (currentTask != null && !currentTask!!.isDone) {
            state.log("generate", "Generation already running; ignoring duplicate request.")
            return
        }
        val prompt = promptField.text
        val temperature = temperatureSlider.value / 100.0f
        val topK = (topKSpinner.value as Number).toInt()
        val maxTokens = (maxTokensSpinner.value as Number).toInt()
        val seed = seedField.text.trim().toLongOrNull() ?: state.defaultSeed

        stopRequested = false
        generateButton.isEnabled = false
        stopButton.isEnabled = true
        outputArea.text = ""
        statusLabel.text = "Generating..."

        val worker = loaded.newWorker()
        val appState = state
        val task = object : SwingWorker<GenerationResult, String>() {
            override fun doInBackground(): GenerationResult {
                return Engine.generateStreaming(
                    loaded = loaded,
                    worker = worker,
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature.toFloat(),
                    topK = topK,
                    seed = seed,
                    shouldStop = { stopRequested || isCancelled },
                    onText = { partial -> publish(partial) }
                )
            }

            override fun process(chunks: MutableList<String>) {
                val latest = chunks.lastOrNull() ?: return
                outputArea.text = latest
                outputArea.caretPosition = outputArea.document.length
            }

            override fun done() {
                generateButton.isEnabled = true
                stopButton.isEnabled = false
                try {
                    val result = get()
                    statusLabel.text = "Done: ${result.generatedTokens} tokens in " +
                        "${Ui.fmtDuration(result.elapsedMs)} (${Ui.fmtFloat(result.tokensPerSec, 1)} tok/s)" +
                        if (result.stoppedOnEos) " — ended on <eos>" else ""
                    appState.log(
                        "generate",
                        "Generated ${result.generatedTokens} tokens (temp=${result.temperature}, topK=${result.topK}, " +
                            "${Ui.fmtFloat(result.tokensPerSec, 1)} tok/s)."
                    )
                } catch (e: Exception) {
                    val cause = e.cause ?: e
                    statusLabel.text = "Generation failed: ${cause.message}"
                    appState.log("generate", "ERROR: ${cause.message}")
                }
            }
        }
        currentTask = task
        task.execute()
    }

    private fun saveOutput() {
        val file = Ui.chooseFile(this, "Save generated text", listOf("txt"), save = true) ?: return
        val target = if (file.name.endsWith(".txt")) file else File(file.path + ".txt")
        target.writeText(outputArea.text)
        state.log("generate", "Saved output to ${target.path}")
        statusLabel.text = "Saved to ${target.name}"
    }
}
