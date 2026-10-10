package forgegpt.gui

import forgegpt.Engine
import forgegpt.Tokenizer
import java.awt.BorderLayout
import java.awt.FlowLayout
import java.awt.GridLayout
import java.io.File
import javax.swing.BorderFactory
import javax.swing.DefaultListModel
import javax.swing.JButton
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JSplitPane
import javax.swing.JTable
import javax.swing.JTextArea
import javax.swing.JTextField
import javax.swing.SwingWorker
import javax.swing.table.DefaultTableModel

/**
 * Corpus tab: dataset statistics for the selected corpus, a tokenizer /
 * vocab inspector, and the bundled public-domain book list.
 *
 * Stats run on a background thread because tokenizing ~1M tokens takes a
 * moment; the inspector tokenizes as you type to show exactly what the
 * model sees (ids, <unk> fallbacks, decode round-trip).
 */
class CorpusPanel(private val state: AppState) : JPanel(BorderLayout()) {

    private val corpusField = JTextField(state.corpusPath)
    private val statsModel = DefaultTableModel(arrayOf("Metric", "Value"), 0)
    private val statsTable = JTable(statsModel)
    private val topTokensModel = DefaultTableModel(arrayOf("Token", "Count"), 0)
    private val topTokensTable = JTable(topTokensModel)
    private val statsStatus = JLabel("Pick a corpus and hit Analyze.")

    private val vocabListModel = DefaultListModel<String>()
    private val vocabList = JList(vocabListModel)
    private val vocabFilter = JTextField()
    private val vocabInfo = JLabel("Load a model/vocab to inspect its tokens.")
    private val inspectInput = JTextArea(4, 40).apply {
        font = Ui.MONO
        lineWrap = true
        wrapStyleWord = true
        text = "Alice was beginning to get very tired of sitting by her sister on the bank."
    }
    private val inspectOutput = JTextArea(6, 40).apply {
        font = Ui.MONO
        isEditable = false
        lineWrap = true
        wrapStyleWord = true
    }

    private val booksModel = DefaultTableModel(arrayOf("Gutenberg", "File", "Title", "Author", "Size"), 0)
    private val booksTable = JTable(booksModel)

    init {
        border = BorderFactory.createEmptyBorder(8, 8, 8, 8)

        val header = JPanel(BorderLayout())
        val pickRow = JPanel(FlowLayout(FlowLayout.LEFT, 6, 0))
        pickRow.add(Ui.label("Corpus:"))
        corpusField.columns = 40
        pickRow.add(corpusField)
        val browse = JButton("Browse...")
        browse.addActionListener {
            Ui.chooseFile(this, "Corpus text", listOf("txt"))?.let { corpusField.text = it.path }
        }
        pickRow.add(browse)
        val analyze = JButton("Analyze")
        analyze.addActionListener { analyzeCorpus() }
        pickRow.add(analyze)
        pickRow.add(statsStatus)
        header.add(pickRow, BorderLayout.NORTH)

        val statsSplit = JSplitPane(
            JSplitPane.HORIZONTAL_SPLIT,
            Ui.titled("Dataset statistics", JScrollPane(statsTable)),
            Ui.titled("Top tokens in corpus", JScrollPane(topTokensTable))
        )
        statsSplit.resizeWeight = 0.55
        header.add(statsSplit, BorderLayout.CENTER)
        add(header, BorderLayout.NORTH)

        val middle = JSplitPane(
            JSplitPane.HORIZONTAL_SPLIT,
            buildVocabInspector(),
            buildTextInspector()
        )
        middle.resizeWeight = 0.45
        add(middle, BorderLayout.CENTER)

        val booksPanel = Ui.titled("Bundled public-domain corpus (./corpus)", JScrollPane(booksTable))
        add(booksPanel, BorderLayout.SOUTH)
        booksTable.rowHeight = 18

        refreshBooks()
    }

    private fun buildVocabInspector(): JPanel {
        val panel = JPanel(BorderLayout())
        panel.border = BorderFactory.createTitledBorder("Vocabulary inspector")
        val top = JPanel(BorderLayout())
        top.add(Ui.label("Filter: "), BorderLayout.WEST)
        top.add(vocabFilter, BorderLayout.CENTER)
        panel.add(top, BorderLayout.NORTH)
        panel.add(JScrollPane(vocabList), BorderLayout.CENTER)
        panel.add(vocabInfo, BorderLayout.SOUTH)

        vocabFilter.addActionListener { refreshVocabList() }
        vocabFilter.document.addDocumentListener(object : javax.swing.event.DocumentListener {
            override fun insertUpdate(e: javax.swing.event.DocumentEvent?) = refreshVocabList()
            override fun removeUpdate(e: javax.swing.event.DocumentEvent?) = refreshVocabList()
            override fun changedUpdate(e: javax.swing.event.DocumentEvent?) = refreshVocabList()
        })
        vocabList.addListSelectionListener { e ->
            if (e.valueIsAdjusting) return@addListSelectionListener
            val tok = vocabList.selectedValue ?: return@addListSelectionListener
            val loaded = state.requireLoaded() ?: return@addListSelectionListener
            val id = loaded.tokenizer.tokenToId[tok]
            vocabInfo.text = if (id != null) "token \"$tok\" -> id $id of ${loaded.tokenizer.vocabSize}" else "not in vocab"
        }
        return panel
    }

    private fun buildTextInspector(): JPanel {
        val panel = JPanel(BorderLayout())
        panel.border = BorderFactory.createTitledBorder("Tokenizer inspector")
        panel.add(Ui.label("Type or paste text — this is exactly what the model sees:"), BorderLayout.NORTH)
        val split = JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            JScrollPane(inspectInput),
            JScrollPane(inspectOutput)
        )
        split.resizeWeight = 0.45
        panel.add(split, BorderLayout.CENTER)
        val run = JButton("Tokenize")
        run.addActionListener { tokenizeInspect() }
        val row = JPanel(FlowLayout(FlowLayout.LEFT, 6, 2))
        row.add(run)
        panel.add(row, BorderLayout.SOUTH)
        return panel
    }

    /** Called after a model/vocab load so the inspector has a tokenizer. */
    fun onModelChanged() {
        refreshVocabList()
        val loaded = state.requireLoaded()
        vocabInfo.text = if (loaded == null) {
            "Load a model/vocab to inspect its tokens."
        } else {
            "Vocab loaded: ${loaded.tokenizer.vocabSize} tokens from ${loaded.vocabFile.name}"
        }
        tokenizeInspect()
    }

    private fun refreshVocabList() {
        val loaded = state.requireLoaded()
        vocabListModel.clear()
        if (loaded == null) return
        val filter = vocabFilter.text.trim().lowercase()
        val all = loaded.tokenizer.idToToken
        val limit = 4000
        var shown = 0
        for ((id, tok) in all.withIndex()) {
            if (filter.isNotEmpty() && !tok.lowercase().contains(filter)) continue
            vocabListModel.addElement(tok)
            shown++
            if (shown >= limit) break
        }
        if (shown == 0 && all.isNotEmpty()) vocabListModel.addElement("(no tokens match)")
    }

    private fun tokenizeInspect() {
        val loaded = state.requireLoaded()
        val text = inspectInput.text
        if (loaded == null) {
            inspectOutput.text = "Load a model/vocab first — the inspector needs its tokenizer."
            return
        }
        val tok = loaded.tokenizer
        val tokenStrings = Tokenizer.tokenizeText(text)
        val ids = tok.encode(text)
        val sb = StringBuilder()
        sb.append("tokens (${ids.size}): ").append(ids.joinToString(" ")).append("\n\n")
        sb.append("as text pieces:\n")
        sb.append(tokenStrings.joinToString(" | ") { "\"$it\"" }).append("\n\n")
        val unkCount = ids.count { it == Tokenizer.UNK }
        sb.append("<unk> count: $unkCount of ${ids.size}")
        if (unkCount > 0) sb.append(" — rare words the model can't see distinctly")
        sb.append("\n\ndecode round-trip:\n")
        sb.append(tok.decode(ids))
        inspectOutput.text = sb.toString()
        inspectOutput.caretPosition = 0
        state.log("corpus", "Inspector: ${ids.size} tokens, $unkCount <unk>.")
    }

    fun refreshBooks() {
        booksModel.rowCount = 0
        val dir = File("corpus")
        for (book in CorpusCatalog.books) {
            val f = File(dir, book.fileName)
            val size = if (f.isFile) Ui.fmtBytes(f.length()) else "missing"
            booksModel.addRow(arrayOf<Any>(book.gutenbergId, book.fileName, book.title, book.author, size))
        }
        val combined = File(dir, CorpusCatalog.COMBINED_FILE)
        if (combined.isFile) {
            booksModel.addRow(
                arrayOf<Any>("—", CorpusCatalog.COMBINED_FILE, "Combined cleaned corpus (what the demo trained on)", "Project Gutenberg", Ui.fmtBytes(combined.length()))
            )
        }
    }

    private fun analyzeCorpus() {
        val path = corpusField.text.trim()
        val file = File(path)
        if (!file.isFile) {
            statsStatus.text = "Not found: $path"
            return
        }
        state.corpusPath = path
        statsStatus.text = "Analyzing ${file.name}..."
        val appState = state
        val task = object : SwingWorker<forgegpt.CorpusStats, Unit>() {
            override fun doInBackground(): forgegpt.CorpusStats {
                val text = file.readText()
                val tokenizer = appState.requireLoaded()?.tokenizer
                    ?: Tokenizer.load(File(appState.vocabPath))
                return Engine.corpusStats(file.name, text, tokenizer)
            }

            override fun done() {
                try {
                    val s = get()
                    statsModel.rowCount = 0
                    fun row(k: String, v: String) = statsModel.addRow(arrayOf<Any>(k, v))
                    row("File", s.sourceName)
                    row("Bytes", Ui.fmtInt(s.bytes) + " (" + Ui.fmtBytes(s.bytes) + ")")
                    row("Characters", Ui.fmtInt(s.chars))
                    row("Lines", Ui.fmtInt(s.lines))
                    row("Tokens (tokenized)", Ui.fmtInt(s.tokens))
                    row("Unique tokens in text", Ui.fmtInt(s.uniqueTokensInText))
                    row("Vocab size (tokenizer)", Ui.fmtInt(s.vocabSize))
                    row("Type/token ratio", Ui.fmtFloat(s.typeTokenRatio))
                    row("<unk> rate vs tokenizer", Ui.fmtFloat(s.unkRate * 100, 2) + " %")
                    row("Avg tokens / line", Ui.fmtFloat(s.avgTokensPerLine, 2))
                    topTokensModel.rowCount = 0
                    for ((tok, count) in s.topTokens) topTokensModel.addRow(arrayOf<Any>(tok, count))
                    statsStatus.text = "Done: ${Ui.fmtInt(s.tokens)} tokens, <unk> ${Ui.fmtFloat(s.unkRate * 100, 2)}%."
                    appState.log("corpus", "Analyzed ${s.sourceName}: ${s.tokens} tokens, unique ${s.uniqueTokensInText}.")
                } catch (e: Exception) {
                    val cause = e.cause ?: e
                    statsStatus.text = "Analyze failed: ${cause.message}"
                    appState.log("corpus", "ERROR analyzing: ${cause.message}")
                }
            }
        }
        task.execute()
    }
}
