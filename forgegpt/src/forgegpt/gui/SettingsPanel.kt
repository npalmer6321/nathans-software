package forgegpt.gui

import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.Insets
import java.io.File
import java.util.Properties
import javax.swing.BorderFactory
import javax.swing.JButton
import javax.swing.JCheckBox
import javax.swing.JPanel
import javax.swing.JTextField

/**
 * Logs / Settings tab.
 *
 * The log half shows every timestamped line the app routes through
 * [AppState.log] (model loads, generation runs, training milestones,
 * errors). The settings half edits the shared paths (model / vocab /
 * corpus), the default generation seed, and output wrapping, persisted
 * to a small properties file next to the working directory so the app
 * remembers them between launches.
 */
class SettingsPanel(private val state: AppState) : JPanel(BorderLayout()) {

    private val logView = LogView("Application log")

    private val modelPathField = JTextField(state.modelPath)
    private val vocabPathField = JTextField(state.vocabPath)
    private val corpusPathField = JTextField(state.corpusPath)
    private val defaultSeedField = JTextField(state.defaultSeed.toString())
    private val wrapOutputBox = JCheckBox("Wrap Generate output", state.wrapOutput)
    private val statusLabel = javax.swing.JLabel("Settings apply immediately; Save writes them to disk.")

    init {
        border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        add(buildSettings(), BorderLayout.NORTH)
        add(logView, BorderLayout.CENTER)

        state.addLogSink { line -> logView.append(line) }
        state.log("app", "Logs tab ready.")
    }

    private fun buildSettings(): JPanel {
        val panel = JPanel(GridBagLayout())
        panel.border = BorderFactory.createTitledBorder("Paths & preferences")
        val c = GridBagConstraints().apply {
            insets = Insets(3, 4, 3, 4)
            anchor = GridBagConstraints.WEST
        }
        var row = 0
        fun addRow(label: String, field: JTextField, browseTitle: String, exts: List<String>) {
            c.gridy = row; c.gridx = 0; c.weightx = 0.0; c.fill = GridBagConstraints.NONE
            panel.add(Ui.label(label), c)
            field.columns = 46
            c.gridx = 1; c.weightx = 1.0; c.fill = GridBagConstraints.HORIZONTAL
            panel.add(field, c)
            val browse = JButton("...")
            browse.addActionListener {
                Ui.chooseFile(this@SettingsPanel, browseTitle, exts)?.let { field.text = it.path }
            }
            c.gridx = 2; c.weightx = 0.0; c.fill = GridBagConstraints.NONE
            panel.add(browse, c)
            row++
        }
        addRow("Model (.fg):", modelPathField, "Model weights", listOf("fg"))
        addRow("Vocab (.txt):", vocabPathField, "Vocab file", listOf("txt"))
        addRow("Corpus (.txt):", corpusPathField, "Corpus text", listOf("txt"))

        c.gridy = row; c.gridx = 0
        panel.add(Ui.label("Default seed:"), c)
        defaultSeedField.columns = 12
        c.gridx = 1; c.weightx = 0.0
        panel.add(defaultSeedField, c)
        c.gridx = 2
        panel.add(wrapOutputBox, c)
        row++

        val buttons = JPanel(FlowLayout(FlowLayout.LEFT, 6, 4))
        val applyButton = JButton("Apply")
        val saveButton = JButton("Save to disk")
        val loadButton = JButton("Load from disk")
        val clearLogButton = JButton("Clear log")
        val copyLogButton = JButton("Copy log")
        buttons.add(applyButton)
        buttons.add(saveButton)
        buttons.add(loadButton)
        buttons.add(clearLogButton)
        buttons.add(copyLogButton)
        buttons.add(statusLabel)
        c.gridy = row; c.gridx = 0; c.gridwidth = 3; c.weightx = 1.0; c.fill = GridBagConstraints.HORIZONTAL
        panel.add(buttons, c)

        applyButton.addActionListener { applyFromFields(); statusLabel.text = "Applied." }
        saveButton.addActionListener {
            applyFromFields()
            saveToDisk()
        }
        loadButton.addActionListener { loadFromDisk() }
        clearLogButton.addActionListener { logView.clear() }
        copyLogButton.addActionListener { logView.copyAll() }
        return panel
    }

    /** Push the field values into shared state. */
    fun applyFromFields() {
        state.modelPath = modelPathField.text.trim().ifEmpty { state.modelPath }
        state.vocabPath = vocabPathField.text.trim().ifEmpty { state.vocabPath }
        state.corpusPath = corpusPathField.text.trim().ifEmpty { state.corpusPath }
        state.defaultSeed = defaultSeedField.text.trim().toLongOrNull() ?: state.defaultSeed
        state.wrapOutput = wrapOutputBox.isSelected
        state.log(
            "settings",
            "Paths: model=${state.modelPath}, vocab=${state.vocabPath}, corpus=${state.corpusPath}, seed=${state.defaultSeed}"
        )
    }

    /** Refresh the fields from shared state (e.g., after training saved new weights). */
    fun refreshFromState() {
        modelPathField.text = state.modelPath
        vocabPathField.text = state.vocabPath
        corpusPathField.text = state.corpusPath
        defaultSeedField.text = state.defaultSeed.toString()
        wrapOutputBox.isSelected = state.wrapOutput
    }

    private fun settingsFile(): File = File("forgegpt-ui.properties")

    private fun saveToDisk() {
        val props = Properties()
        props.setProperty("modelPath", state.modelPath)
        props.setProperty("vocabPath", state.vocabPath)
        props.setProperty("corpusPath", state.corpusPath)
        props.setProperty("defaultSeed", state.defaultSeed.toString())
        props.setProperty("wrapOutput", state.wrapOutput.toString())
        settingsFile().outputStream().use { out ->
            props.store(out, "ForgeGPT GUI settings")
        }
        statusLabel.text = "Saved to ${settingsFile().path}"
        state.log("settings", "Saved settings to ${settingsFile().path}")
    }

    private fun loadFromDisk() {
        val file = settingsFile()
        if (!file.isFile) {
            statusLabel.text = "No saved settings at ${file.path} yet."
            return
        }
        val props = Properties()
        file.inputStream().use { props.load(it) }
        state.modelPath = props.getProperty("modelPath", state.modelPath)
        state.vocabPath = props.getProperty("vocabPath", state.vocabPath)
        state.corpusPath = props.getProperty("corpusPath", state.corpusPath)
        state.defaultSeed = props.getProperty("defaultSeed")?.toLongOrNull() ?: state.defaultSeed
        state.wrapOutput = props.getProperty("wrapOutput")?.toBoolean() ?: state.wrapOutput
        refreshFromState()
        statusLabel.text = "Loaded settings from ${file.path}"
        state.log("settings", "Loaded settings from ${file.path}")
    }

    /** Load persisted settings at startup if the file exists. */
    fun loadPersistedAtStartup() {
        if (settingsFile().isFile) loadFromDisk()
    }
}
