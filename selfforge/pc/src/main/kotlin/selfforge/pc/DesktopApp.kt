package selfforge.pc

import selfforge.Example
import selfforge.SelfTrainer
import selfforge.Skill
import selfforge.SkillRegistry
import selfforge.codegen.CharRNN
import selfforge.codegen.CodeCorpus
import selfforge.codegen.CodeSelfTrainer
import selfforge.codegen.TinyPy
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.FlowLayout
import java.awt.Font
import java.io.File
import javax.swing.*
import kotlin.concurrent.thread
import kotlin.math.sqrt
import kotlin.random.Random

fun main() = SwingUtilities.invokeLater { DesktopApp() }

/**
 * SelfForge for PC — the same self-teaching AI as the Android app,
 * as a desktop program. Needs Java 17+.
 *
 * Two tabs:
 *  - Patterns: teach it inside-vs-outside a circle, then let it teach itself.
 *  - Code: it learns to write tiny programs, runs them, keeps the ones that
 *    work, and gets better each round.
 */
class DesktopApp : JFrame("SelfForge — an AI that teaches itself") {

    private val registry = SkillRegistry()
    private var skill: Skill = newCircleSkill()

    private var codeRnn: CharRNN? = null
    private var lastProgram = ""
    private val codeInterp = TinyPy()

    private val patternLog = makeLogArea()
    private val codeLog = makeLogArea()
    private val askResult = JLabel(" ")

    init {
        defaultCloseOperation = EXIT_ON_CLOSE
        val tabs = JTabbedPane()
        tabs.addTab("Patterns", buildPatternsPanel())
        tabs.addTab("Code", buildCodePanel())
        add(tabs)
        preferredSize = Dimension(780, 880)
        pack()
        setLocationRelativeTo(null)
        isVisible = true
    }

    private fun newCircleSkill(): Skill =
        registry.newSkill("circle", 2, listOf("inside", "outside"), hiddenSize = 24, seed = System.currentTimeMillis())
            .also { it.net.learningRate = 0.1 }

    private fun circleLabel(p: DoubleArray): Int =
        if (sqrt(p[0] * p[0] + p[1] * p[1]) < 0.55) 0 else 1

    // ================= Patterns tab =================

    private fun buildPatternsPanel(): JPanel {
        val panel = JPanel(BorderLayout(8, 8))
        panel.border = BorderFactory.createEmptyBorder(12, 12, 12, 12)

        val controls = JPanel()
        controls.layout = BoxLayout(controls, BoxLayout.Y_AXIS)
        controls.add(introLabel(
            "<html><b>Can it learn to see an invisible circle?</b><br>" +
            "Give it examples of points that are inside or outside the circle. " +
            "Then quiz it — and afterwards, let it teach <i>itself</i> from new points you never labeled.</html>"
        ))

        val demoBtn = JButton("▶  See it learn")
        demoBtn.addActionListener { runBg(demoBtn) { runPatternDemo() } }
        controls.add(demoBtn)
        controls.add(Box.createVerticalStrut(6))

        controls.add(sectionLabel("Teach it"))
        controls.add(hintLabel("Type a point, say inside or outside, and teach it. Try (0, 0) → inside, (0.9, 0.9) → outside."))
        val teachX = numField("x"); val teachY = numField("y")
        val spinner = JComboBox(arrayOf("inside", "outside"))
        val teachBtn = JButton("Teach")
        teachBtn.addActionListener {
            val x = teachX.text.toDoubleOrNull()
            val y = teachY.text.toDoubleOrNull()
            if (x == null || y == null) { plog("Type numbers for x and y first."); return@addActionListener }
            val label = spinner.selectedItem as String
            skill.teach(doubleArrayOf(x, y), label)
            plog("Learned: ($x, $y) is $label.  (${skill.examples.size} examples so far)")
        }
        controls.add(row(teachX, teachY, spinner, teachBtn))

        controls.add(sectionLabel("Train it"))
        val trainBtn = JButton("Train")
        val selfBtn = JButton("Let it teach itself")
        trainBtn.addActionListener {
            runBg(trainBtn, selfBtn) {
                if (skill.examples.isEmpty()) { plog("Teach it some examples first (above)."); return@runBg }
                plog("Training on your ${skill.examples.size} examples...")
                skill.train(800)
                plog("Done — it now gets ${pct(circleTestAccuracy(150))} of 150 test points right.")
            }
        }
        selfBtn.addActionListener {
            runBg(trainBtn, selfBtn) {
                if (skill.examples.isEmpty()) { plog("Teach it some examples first (above)."); return@runBg }
                plog("Self-teaching: it will guess 200 new points itself and retrain on the sure ones.")
                val rng = Random(System.currentTimeMillis())
                val unlabeled = (1..200).map { doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)) }
                val validation = (1..150).map {
                    val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
                    Example(p, circleLabel(p))
                }
                val before = skill.accuracy(validation)
                val reports = skill.selfTrain(unlabeled, validation, SelfTrainer(0.85, 6, 200, 3))
                for (r in reports) plog("  round ${r.round}: ${pct(r.valAccuracy)} — labeled ${r.pseudoLabeledAdded} points itself")
                plog("Self-teaching done: ${pct(before)} → ${pct(skill.accuracy(validation))}, with zero new labels from you.")
            }
        }
        controls.add(row(trainBtn, selfBtn))

        controls.add(sectionLabel("Quiz it"))
        val askX = numField("x"); val askY = numField("y")
        val askBtn = JButton("Where is it?")
        askBtn.addActionListener {
            val x = askX.text.toDoubleOrNull()
            val y = askY.text.toDoubleOrNull()
            if (x == null || y == null) { plog("Type numbers for x and y first."); return@addActionListener }
            val (label, conf) = skill.guessWithConfidence(doubleArrayOf(x, y))
            askResult.text = "$label (${pct(conf)})"
            plog("You asked about ($x, $y) → it says $label, ${pct(conf)} sure.")
        }
        controls.add(row(askX, askY, askBtn, askResult))

        val saveBtn = JButton("Save")
        val loadBtn = JButton("Load")
        val resetBtn = JButton("Start over")
        val skillFile = File("selfforge-skill.txt")
        saveBtn.addActionListener {
            skillFile.writeText(skill.save())
            plog("Saved its brain (${skill.net.weightCount()} weights).")
        }
        loadBtn.addActionListener {
            if (!skillFile.exists()) { plog("Nothing saved yet."); return@addActionListener }
            registry.remove(skill.name)
            skill = Skill.load(skillFile.readText())
            plog("Loaded its saved brain.")
        }
        resetBtn.addActionListener {
            registry.remove(skill.name)
            skill = newCircleSkill()
            plog("Fresh start — it forgot everything. Teach it again!")
        }
        controls.add(row(saveBtn, loadBtn, resetBtn))

        panel.add(controls, BorderLayout.NORTH)
        patternLog.text = "Ready. Click “▶ See it learn”, or teach it yourself above.\n"
        panel.add(JScrollPane(patternLog), BorderLayout.CENTER)
        return panel
    }

    // ================= Code tab =================

    private fun buildCodePanel(): JPanel {
        val panel = JPanel(BorderLayout(8, 8))
        panel.border = BorderFactory.createEmptyBorder(12, 12, 12, 12)

        val controls = JPanel()
        controls.layout = BoxLayout(controls, BoxLayout.Y_AXIS)
        controls.add(introLabel(
            "<html><b>It learns to write tiny programs — then actually runs them.</b><br>" +
            "Each round it writes new programs, runs each one, keeps the ones that " +
            "work, and studies them. Watch the share of working programs climb.</html>"
        ))

        controls.add(sectionLabel("Its little language"))
        val guide = JTextArea(
            "x = 3 + 4        give x the value 7\n" +
            "print(x * 2)     show 14\n" +
            "if x > 2:        if true, do the indented lines\n" +
            "    y = 99\n" +
            "while x > 0:     repeat while true\n" +
            "    x = x - 1"
        )
        guide.isEditable = false
        guide.font = Font(Font.MONOSPACED, Font.PLAIN, 12)
        guide.background = panel.background.brighter()
        guide.border = BorderFactory.createEmptyBorder(8, 8, 8, 8)
        controls.add(guide)
        controls.add(Box.createVerticalStrut(6))

        val demoBtn = JButton("▶  See it learn to code")
        demoBtn.addActionListener { runBg(demoBtn) { runCodeDemo() } }
        controls.add(demoBtn)
        controls.add(Box.createVerticalStrut(6))

        val trainBtn = JButton("Study examples")
        val selfBtn = JButton("Self-teach loop")
        trainBtn.addActionListener {
            runBg(trainBtn, selfBtn) {
                clog("Studying ${CodeCorpus.programs.size} example programs... (takes about a minute)")
                val rnn = CharRNN(CodeCorpus.vocab, hiddenSize = 128, learningRate = 0.05, seed = 7L)
                rnn.train(CodeCorpus.programs, epochs = 250)
                codeRnn = rnn
                clog("Done studying (${rnn.weightCount()} weights). Now run the self-teach loop!")
            }
        }
        selfBtn.addActionListener {
            runBg(trainBtn, selfBtn) {
                val rnn = codeRnn ?: run { clog("Click “Study examples” first."); return@runBg }
                clog("Self-teaching: writing programs, running them, keeping the ones that work...")
                val corpus = CodeCorpus.programs.toMutableList()
                val reports = CodeSelfTrainer(40, 6, 60, 0.8).selfTeach(
                    rnn, corpus, codeInterp, System.currentTimeMillis())
                for (r in reports)
                    clog("  round ${r.round}: ${"%.0f".format(r.validPct * 100)}% of its programs ran — kept ${r.kept}")
                clog("Done — its library grew to ${corpus.size} programs, most written by itself.")
            }
        }
        controls.add(row(trainBtn, selfBtn))

        val writeBtn = JButton("Write a program")
        val runBtn = JButton("Run it")
        writeBtn.addActionListener {
            val rnn = codeRnn ?: run { clog("Click “Study examples” first."); return@addActionListener }
            val rng = Random(System.currentTimeMillis())
            val seedLine = CodeCorpus.programs.random(rng).lineSequence().first()
            lastProgram = rnn.sample(seedLine + "\n", 140, 0.8, rng).trim() + "\n"
            clog("— it wrote this: —\n$lastProgram")
        }
        runBtn.addActionListener {
            if (lastProgram.isEmpty()) { clog("Click “Write a program” first."); return@addActionListener }
            when (val res = codeInterp.run(lastProgram)) {
                is TinyPy.Result.Ok -> clog("It runs! Output: ${res.output}")
                is TinyPy.Result.Err -> clog("It crashed: ${res.message}")
            }
        }
        controls.add(row(writeBtn, runBtn))

        panel.add(controls, BorderLayout.NORTH)
        codeLog.text = "Welcome to the Code Lab! Click “▶ See it learn to code” for the full show.\n"
        panel.add(JScrollPane(codeLog), BorderLayout.CENTER)
        return panel
    }

    // ================= scripted demos =================

    private fun runPatternDemo() {
        registry.remove(skill.name)
        skill = newCircleSkill()
        val presets = listOf(
            doubleArrayOf(0.0, 0.0) to "inside", doubleArrayOf(0.3, 0.2) to "inside",
            doubleArrayOf(-0.4, 0.1) to "inside", doubleArrayOf(0.2, -0.35) to "inside",
            doubleArrayOf(0.1, 0.5) to "inside", doubleArrayOf(-0.2, 0.45) to "inside",
            doubleArrayOf(0.9, 0.9) to "outside", doubleArrayOf(-0.8, -0.7) to "outside",
            doubleArrayOf(0.7, 0.0) to "outside", doubleArrayOf(0.0, 0.8) to "outside",
            doubleArrayOf(0.4, 0.4) to "outside", doubleArrayOf(-0.5, 0.3) to "outside"
        )
        for ((p, l) in presets) skill.teach(p, l)
        plog("I showed it 12 examples — 6 inside the circle, 6 outside.")
        plog("Training...")
        skill.train(800)
        plog("It now gets ${pct(circleTestAccuracy(150))} of 150 test points right.")
        plog("Now the fun part: it teaches ITSELF from 200 points I never labeled.")
        val rng = Random(42L)
        val unlabeled = (1..200).map { doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)) }
        val validation = (1..150).map {
            val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
            Example(p, circleLabel(p))
        }
        val before = skill.accuracy(validation)
        val reports = skill.selfTrain(unlabeled, validation, SelfTrainer(0.85, 6, 200, 3))
        for (r in reports) plog("  round ${r.round}: ${pct(r.valAccuracy)} — it labeled ${r.pseudoLabeledAdded} points itself")
        plog("Finished: ${pct(before)} → ${pct(skill.accuracy(validation))}. Now quiz it yourself above!")
    }

    private fun runCodeDemo() {
        clog("Step 1: studying ${CodeCorpus.programs.size} example programs... (about a minute)")
        val rnn = CharRNN(CodeCorpus.vocab, hiddenSize = 128, learningRate = 0.05, seed = 7L)
        rnn.train(CodeCorpus.programs, epochs = 250)
        codeRnn = rnn
        clog("Step 2: now it teaches itself — writing, running, keeping what works.")
        val corpus = CodeCorpus.programs.toMutableList()
        val reports = CodeSelfTrainer(40, 6, 60, 0.8).selfTeach(rnn, corpus, codeInterp, System.currentTimeMillis())
        for (r in reports)
            clog("  round ${r.round}: ${"%.0f".format(r.validPct * 100)}% ran — kept ${r.kept} new programs")
        val rng = Random(System.currentTimeMillis())
        val seedLine = CodeCorpus.programs.random(rng).lineSequence().first()
        lastProgram = rnn.sample(seedLine + "\n", 140, 0.8, rng).trim() + "\n"
        clog("Step 3: here's one it wrote all by itself:\n$lastProgram")
        when (val res = codeInterp.run(lastProgram)) {
            is TinyPy.Result.Ok -> clog("And it runs! Output: ${res.output}")
            is TinyPy.Result.Err -> clog("This one crashed (${res.message}) — click “Write a program” for another.")
        }
    }

    // ================= helpers =================

    private fun circleTestAccuracy(n: Int): Double {
        val rng = Random(99L)
        return skill.accuracy((1..n).map {
            val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
            Example(p, circleLabel(p))
        })
    }

    private fun pct(d: Double) = "${"%.0f".format(d * 100)}%"

    private fun introLabel(html: String) = JLabel(html).apply {
        border = BorderFactory.createEmptyBorder(4, 4, 10, 4)
    }

    private fun sectionLabel(t: String) = JLabel(t).apply {
        font = font.deriveFont(Font.BOLD, 14f)
        border = BorderFactory.createEmptyBorder(10, 0, 2, 0)
    }

    private fun hintLabel(t: String) = JLabel("<html><font color='#777777'>$t</font></html>")

    private fun numField(h: String) = JTextField(6).apply { toolTipText = h }

    private fun row(vararg comps: JComponent) = JPanel(FlowLayout(FlowLayout.LEFT, 6, 2)).apply {
        for (c in comps) add(c)
    }

    private fun makeLogArea() = JTextArea(14, 70).apply {
        isEditable = false
        font = Font(Font.MONOSPACED, Font.PLAIN, 12)
    }

    private fun plog(msg: String) = SwingUtilities.invokeLater {
        patternLog.append(msg + "\n")
        patternLog.caretPosition = patternLog.document.length
    }

    private fun clog(msg: String) = SwingUtilities.invokeLater {
        codeLog.append(msg + "\n")
        codeLog.caretPosition = codeLog.document.length
    }

    private fun runBg(vararg buttons: JButton, work: () -> Unit) {
        buttons.forEach { it.isEnabled = false }
        thread {
            try { work() } catch (e: Exception) { plog("Oops, something went wrong: ${e.message}") }
            finally { SwingUtilities.invokeLater { buttons.forEach { it.isEnabled = true } } }
        }
    }
}
