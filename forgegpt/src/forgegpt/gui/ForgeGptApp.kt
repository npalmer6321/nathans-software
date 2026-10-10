package forgegpt.gui

import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.io.File
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JTabbedPane
import javax.swing.JTextField
import javax.swing.SwingUtilities

/**
 * ForgeGptApp — the main window.
 *
 * Top bar: model/vocab path pickers + Load button + status line, shared
 * by all tabs. Below: five tabs (Generate, Train, Corpus, Model Lab,
 * Logs/Settings). Loading a model notifies every panel so inspectors and
 * readouts refresh against the same [AppState.loaded] instance.
 */
class ForgeGptApp : JFrame("ForgeGPT — from-scratch transformer playground") {

    private val state = AppState()

    private val modelPathField = JTextField(state.modelPath)
    private val vocabPathField = JTextField(state.vocabPath)
    private val loadButton = JButton("Load model")
    private val statusLabel = JLabel("Model not loaded.")

    private val generatePanel = GeneratePanel(state)
    private val trainPanel = TrainPanel(state)
    private val corpusPanel = CorpusPanel(state)
    private val modelLabPanel = ModelLabPanel(state)
    private val settingsPanel = SettingsPanel(state)

    private val tabs = JTabbedPane()

    init {
        defaultCloseOperation = EXIT_ON_CLOSE
        layout = BorderLayout()
        add(buildTopBar(), BorderLayout.NORTH)

        tabs.addTab("Generate", generatePanel)
        tabs.addTab("Train", trainPanel)
        tabs.addTab("Corpus", corpusPanel)
        tabs.addTab("Model Lab", modelLabPanel)
        tabs.addTab("Logs / Settings", settingsPanel)
        add(tabs, BorderLayout.CENTER)

        minimumSize = Dimension(880, 620)
        setSize(1080, 760)
        setLocationRelativeTo(null)

        settingsPanel.loadPersistedAtStartup()
        syncTopBarFromState()
        state.addLogSink { line -> updateStatusFromLog(line) }
        state.log("app", "ForgeGPT GUI started. Working dir: ${File(".").absolutePath}")
    }

    private fun buildTopBar(): JPanel {
        val bar = JPanel(BorderLayout())
        bar.border = BorderFactory.createEmptyBorder(6, 8, 2, 8)
        val fields = JPanel(FlowLayout(FlowLayout.LEFT, 6, 2))
        fields.add(Ui.label("Model:"))
        modelPathField.columns = 26
        fields.add(modelPathField)
        val browseModel = JButton("...")
        browseModel.addActionListener {
            Ui.chooseFile(this, "Model weights (.fg)", listOf("fg"))?.let { modelPathField.text = it.path }
        }
        fields.add(browseModel)
        fields.add(Ui.label("Vocab:"))
        vocabPathField.columns = 22
        fields.add(vocabPathField)
        val browseVocab = JButton("...")
        browseVocab.addActionListener {
            Ui.chooseFile(this, "Vocab file (.txt)", listOf("txt"))?.let { vocabPathField.text = it.path }
        }
        fields.add(browseVocab)
        fields.add(loadButton)
        bar.add(fields, BorderLayout.NORTH)
        statusLabel.font = Ui.LABEL_FONT
        bar.add(statusLabel, BorderLayout.SOUTH)

        loadButton.addActionListener { loadModel() }
        return bar
    }

    private fun syncTopBarFromState() {
        modelPathField.text = state.modelPath
        vocabPathField.text = state.vocabPath
    }

    private fun updateStatusFromLog(line: String) {
        // Mirror the most recent model-related line into the header status.
        if (line.contains("[model]") || line.contains("[settings]")) {
            SwingUtilities.invokeLater {
                statusLabel.text = line.substringAfter("] ", line)
            }
        }
    }

    /** Load the model at the top-bar paths and broadcast to all tabs. */
    fun loadModel() {
        state.modelPath = modelPathField.text.trim().ifEmpty { state.modelPath }
        state.vocabPath = vocabPathField.text.trim().ifEmpty { state.vocabPath }
        statusLabel.text = "Loading ${state.modelPath}..."
        try {
            val loaded = state.loadModelFromPaths()
            statusLabel.text = "Ready: ${loaded.describe()}"
            generatePanel.onModelChanged()
            corpusPanel.onModelChanged()
            modelLabPanel.onModelChanged()
            settingsPanel.refreshFromState()
        } catch (e: Exception) {
            statusLabel.text = "Load failed: ${e.message}"
            state.log("model", "ERROR: ${e.message}")
            javax.swing.JOptionPane.showMessageDialog(
                this,
                "Could not load model:\n${e.message}",
                "Load failed",
                javax.swing.JOptionPane.ERROR_MESSAGE
            )
        }
    }

    companion object {
        /** Build and show the window. Must be called on the EDT. */
        fun launch() {
            val app = ForgeGptApp()
            app.isVisible = true
            // Convenience: if the default demo files are present, load them right away.
            if (File(app.state.modelPath).isFile && File(app.state.vocabPath).isFile) {
                app.loadModel()
            }
        }
    }
}
