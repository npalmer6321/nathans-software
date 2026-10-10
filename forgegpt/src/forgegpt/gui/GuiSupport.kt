package forgegpt.gui

import forgegpt.Engine
import forgegpt.LoadedModel
import java.awt.BorderLayout
import java.awt.Component
import java.awt.Dimension
import java.awt.Font
import java.awt.Toolkit
import java.awt.datatransfer.StringSelection
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import javax.swing.BorderFactory
import javax.swing.JFileChooser
import javax.swing.JLabel
import javax.swing.JPanel
import javax.swing.JScrollPane
import javax.swing.JTextArea
import javax.swing.SwingUtilities
import javax.swing.filechooser.FileNameExtensionFilter

/**
 * Shared GUI support: application state, a thread-safe log view, small
 * formatting helpers, and file-picker shortcuts. Every panel talks to
 * the rest of the app through [AppState] so model loads, log lines and
 * settings stay consistent across tabs.
 */

/** Mutable app-wide state shared by all tabs. */
class AppState {
    @Volatile
    var loaded: LoadedModel? = null

    @Volatile
    var modelPath: String = "model.fg"

    @Volatile
    var vocabPath: String = "vocab.txt"

    @Volatile
    var corpusPath: String = "corpus/corpus.txt"

    /** Default sampling seed for the Generate tab. */
    @Volatile
    var defaultSeed: Long = 7L

    /** Whether the Generate tab wraps its output area. */
    @Volatile
    var wrapOutput: Boolean = true

    /** Registered log sinks (the Logs tab, status mirrors, ...). */
    private val sinks = ArrayList<(String) -> Unit>()

    fun addLogSink(sink: (String) -> Unit) {
        synchronized(sinks) { sinks.add(sink) }
    }

    /** Append a timestamped line to every registered log sink. Safe from any thread. */
    fun log(source: String, message: String) {
        val stamp = SimpleDateFormat("HH:mm:ss").format(Date())
        val line = "[$stamp] [$source] $message"
        val targets: List<(String) -> Unit>
        synchronized(sinks) { targets = ArrayList(sinks) }
        for (sink in targets) {
            try {
                sink(line)
            } catch (t: Throwable) {
                System.err.println("log sink failed: ${t.message}")
            }
        }
    }

    /**
     * Load (or reload) the model at the current paths. Returns the loaded
     * model or throws with a user-readable message.
     */
    fun loadModelFromPaths(): LoadedModel {
        val m = Engine.loadModel(File(modelPath), File(vocabPath))
        loaded = m
        log("model", "Loaded ${m.describe()} from $modelPath")
        return m
    }

    /** The loaded model, or null if none has been loaded yet. */
    fun requireLoaded(): LoadedModel? = loaded
}

/** A scrollable, append-only log area that is safe to write from any thread. */
class LogView(title: String) : JPanel(BorderLayout()) {
    private val area = JTextArea().apply {
        isEditable = false
        font = Font(Font.MONOSPACED, Font.PLAIN, 12)
        lineWrap = false
    }
    private val scroll = JScrollPane(area)

    init {
        border = BorderFactory.createTitledBorder(title)
        add(scroll, BorderLayout.CENTER)
        preferredSize = Dimension(640, 160)
    }

    fun append(line: String) {
        SwingUtilities.invokeLater {
            area.append(line + "\n")
            // keep the buffer bounded so a long training run can't eat the heap
            val maxChars = 200_000
            val doc = area.document
            if (doc.length > maxChars) {
                try {
                    doc.remove(0, doc.length - maxChars)
                } catch (ignored: Exception) {
                    // racing with an edit; next append will retry the trim
                }
            }
            area.caretPosition = doc.length
        }
    }

    fun clear() {
        SwingUtilities.invokeLater { area.text = "" }
    }

    fun text(): String = area.text

    fun copyAll() {
        val sel = StringSelection(area.text)
        Toolkit.getDefaultToolkit().systemClipboard.setContents(sel, null)
    }
}

/** Small layout/formatting helpers shared by the panels. */
object Ui {
    val MONO: Font = Font(Font.MONOSPACED, Font.PLAIN, 12)
    val MONO_BOLD: Font = Font(Font.MONOSPACED, Font.BOLD, 12)
    val LABEL_FONT: Font = Font(Font.SANS_SERIF, Font.PLAIN, 12)

    fun titled(title: String, comp: Component): JPanel {
        val p = JPanel(BorderLayout())
        p.border = BorderFactory.createTitledBorder(title)
        p.add(comp, BorderLayout.CENTER)
        return p
    }

    fun label(text: String): JLabel = JLabel(text).apply { font = LABEL_FONT }

    fun fmtInt(n: Long): String = "%,d".format(n)
    fun fmtInt(n: Int): String = "%,d".format(n)

    fun fmtFloat(x: Double, digits: Int = 4): String = "%.${digits}f".format(x)
    fun fmtFloat(x: Float, digits: Int = 4): String = "%.${digits}f".format(x)

    fun fmtBytes(bytes: Long): String {
        if (bytes < 1024) return "$bytes B"
        val units = arrayOf("KB", "MB", "GB")
        var v = bytes.toDouble() / 1024.0
        var u = 0
        while (v >= 1024 && u < units.size - 1) {
            v /= 1024.0
            u++
        }
        return "%.1f %s".format(v, units[u])
    }

    fun fmtDuration(ms: Long): String {
        if (ms < 1000) return "$ms ms"
        val s = ms / 1000.0
        if (s < 60) return "%.1f s".format(s)
        val m = (s / 60).toInt()
        val rem = s - m * 60
        return "%d m %.0f s".format(m, rem)
    }

    /** Run [block] on the EDT and return its result (rethrows on failure). */
    fun <T> onEdt(block: () -> T): T {
        if (SwingUtilities.isEventDispatchThread()) return block()
        var result: T? = null
        var failure: Throwable? = null
        SwingUtilities.invokeAndWait {
            try {
                result = block()
            } catch (t: Throwable) {
                failure = t
            }
        }
        failure?.let { throw it }
        @Suppress("UNCHECKED_CAST")
        return result as T
    }

    fun chooseFile(parent: Component?, title: String, extensions: List<String>, save: Boolean = false): File? {
        val chooser = JFileChooser().apply {
            dialogTitle = title
            if (extensions.isNotEmpty()) {
                fileFilter = FileNameExtensionFilter(extensions.joinToString(", "), *extensions.toTypedArray())
            }
            val dir = File(System.getProperty("user.dir"))
            if (dir.isDirectory) currentDirectory = dir
        }
        val result = if (save) chooser.showSaveDialog(parent) else chooser.showOpenDialog(parent)
        return if (result == JFileChooser.APPROVE_OPTION) chooser.selectedFile else null
    }

    fun chooseDirectory(parent: Component?, title: String): File? {
        val chooser = JFileChooser().apply {
            dialogTitle = title
            fileSelectionMode = JFileChooser.DIRECTORIES_ONLY
            val dir = File(System.getProperty("user.dir"))
            if (dir.isDirectory) currentDirectory = dir
        }
        return if (chooser.showOpenDialog(parent) == JFileChooser.APPROVE_OPTION) chooser.selectedFile else null
    }
}

/** Public-domain books bundled with ForgeGPT, with Gutenberg ids. */
data class CorpusBook(val gutenbergId: Int, val fileName: String, val title: String, val author: String)

object CorpusCatalog {
    val books: List<CorpusBook> = listOf(
        CorpusBook(11, "pg11.txt", "Alice's Adventures in Wonderland", "Lewis Carroll"),
        CorpusBook(1342, "pg1342.txt", "Pride and Prejudice", "Jane Austen"),
        CorpusBook(1661, "pg1661.txt", "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
        CorpusBook(84, "pg84.txt", "Frankenstein", "Mary Shelley"),
        CorpusBook(98, "pg98.txt", "A Tale of Two Cities", "Charles Dickens"),
        CorpusBook(2701, "pg2701.txt", "Moby Dick", "Herman Melville")
    )

    /** The combined, cleaned corpus the shipped model trained on. */
    const val COMBINED_FILE = "corpus.txt"
}
