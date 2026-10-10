package com.selfforge.app

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.*
import selfforge.Example
import selfforge.SelfTrainer
import selfforge.Skill
import selfforge.SkillRegistry
import selfforge.codegen.CharRNN
import selfforge.codegen.CodeCorpus
import selfforge.codegen.CodeSelfTrainer
import selfforge.codegen.TinyPy
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * SelfForge v1.2 — a friendlier face on a self-teaching AI.
 *
 * Two labs, plain language throughout:
 *  - PATTERNS: teach it inside-vs-outside a circle, then let it teach itself.
 *  - CODE: it learns to write tiny programs, runs them, keeps the ones that
 *    work, and gets better each round.
 */
class MainActivity : Activity() {

    // ---------- state ----------
    private val registry = SkillRegistry()
    private var skill: Skill = newCircleSkill()

    private var codeRnn: CharRNN? = null
    private var lastProgram: String = ""
    private val codeInterp = TinyPy()

    // ---------- views ----------
    private lateinit var tabPatternsBtn: Button
    private lateinit var tabCodeBtn: Button
    private lateinit var patternsPane: LinearLayout
    private lateinit var codePane: LinearLayout
    private lateinit var patternLog: TextView
    private lateinit var patternScroll: ScrollView
    private lateinit var codeLog: TextView
    private lateinit var codeScroll: ScrollView
    private lateinit var askResult: TextView

    private fun newCircleSkill(): Skill =
        registry.newSkill("circle", 2, listOf("inside", "outside"), hiddenSize = 24, seed = System.currentTimeMillis())
            .also { it.net.learningRate = 0.1 }

    private fun circleLabel(p: DoubleArray): Int =
        if (sqrt(p[0] * p[0] + p[1] * p[1]) < 0.55) 0 else 1

    // ================= UI construction =================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 28, 28, 28)
        }

        root.addView(title("SelfForge", 24f))
        root.addView(subtitle("An AI that teaches itself — right on your phone."))

        // Tabs
        tabPatternsBtn = tabButton("Patterns")
        tabCodeBtn = tabButton("Code")
        tabPatternsBtn.setOnClickListener { showTab(true) }
        tabCodeBtn.setOnClickListener { showTab(false) }
        root.addView(row(tabPatternsBtn, tabCodeBtn))

        patternsPane = buildPatternsPane()
        codePane = buildCodePane()
        codePane.visibility = View.GONE
        root.addView(patternsPane)
        root.addView(codePane)
        showTab(true)

        setContentView(root)
    }

    private fun showTab(patterns: Boolean) {
        patternsPane.visibility = if (patterns) View.VISIBLE else View.GONE
        codePane.visibility = if (patterns) View.GONE else View.VISIBLE
        styleTab(tabPatternsBtn, patterns)
        styleTab(tabCodeBtn, !patterns)
    }

    private fun styleTab(b: Button, selected: Boolean) {
        if (selected) {
            b.setBackgroundColor(0xFF1A73E8.toInt())
            b.setTextColor(0xFFFFFFFF.toInt())
        } else {
            b.setBackgroundColor(0xFFE8E8E8.toInt())
            b.setTextColor(0xFF333333.toInt())
        }
    }

    // ---------- Patterns pane ----------

    private fun buildPatternsPane(): LinearLayout {
        val pane = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
        }

        pane.addView(card(
            "Can it learn to see an invisible circle?\n\n" +
            "Give it examples of points that are inside or outside the circle. " +
            "Then quiz it — and afterwards, let it teach ITSELF from new points you never labeled."
        ))

        val demoBtn = Button(this).apply { text = "▶  See it learn" }
        demoBtn.setOnClickListener { runInBackground(demoBtn) { runPatternDemo() } }
        pane.addView(demoBtn)

        pane.addView(section("Teach it"))
        pane.addView(hint("Tap in a point, say inside or outside, and teach it. Try (0, 0) → inside, (0.9, 0.9) → outside."))
        val teachX = numInput("x"); val teachY = numInput("y")
        val spinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity,
                android.R.layout.simple_spinner_item, listOf("inside", "outside")).also {
                it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
        }
        val teachBtn = Button(this).apply { text = "Teach" }
        teachBtn.setOnClickListener {
            val x = teachX.text.toString().toDoubleOrNull()
            val y = teachY.text.toString().toDoubleOrNull()
            if (x == null || y == null) { plog("Type numbers for x and y first."); return@setOnClickListener }
            val label = spinner.selectedItem as String
            skill.teach(doubleArrayOf(x, y), label)
            plog("Learned: ($x, $y) is $label.  (${skill.examples.size} examples so far)")
        }
        pane.addView(row(teachX, teachY, spinner, teachBtn))

        pane.addView(section("Train it"))
        val trainBtn = Button(this).apply { text = "Train" }
        val selfBtn = Button(this).apply { text = "Let it teach itself" }
        trainBtn.setOnClickListener {
            runInBackground(trainBtn, selfBtn) {
                if (skill.examples.isEmpty()) { plog("Teach it some examples first (above)."); return@runInBackground }
                plog("Training on your ${skill.examples.size} examples...")
                skill.train(800)
                plog("Done — it now gets ${pct(circleTestAccuracy(150))} of 150 test points right.")
            }
        }
        selfBtn.setOnClickListener {
            runInBackground(trainBtn, selfBtn) {
                if (skill.examples.isEmpty()) { plog("Teach it some examples first (above)."); return@runInBackground }
                plog("Self-teaching: it will guess 200 new points itself and retrain on the sure ones.")
                val rng = Random(System.currentTimeMillis())
                val unlabeled = (1..200).map { doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)) }
                val validation = (1..150).map { val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)); Example(p, circleLabel(p)) }
                val before = skill.accuracy(validation)
                val reports = skill.selfTrain(unlabeled, validation, SelfTrainer(0.85, 6, 200, 3))
                for (r in reports) plog("  round ${r.round}: ${pct(r.valAccuracy)} — labeled ${r.pseudoLabeledAdded} points itself")
                plog("Self-teaching done: ${pct(before)} → ${pct(skill.accuracy(validation))}, with zero new labels from you.")
            }
        }
        pane.addView(row(trainBtn, selfBtn))

        pane.addView(section("Quiz it"))
        val askX = numInput("x"); val askY = numInput("y")
        val askBtn = Button(this).apply { text = "Where is it?" }
        askResult = TextView(this).apply { textSize = 16f; setPadding(8, 0, 0, 0) }
        askBtn.setOnClickListener {
            val x = askX.text.toString().toDoubleOrNull()
            val y = askY.text.toString().toDoubleOrNull()
            if (x == null || y == null) { plog("Type numbers for x and y first."); return@setOnClickListener }
            val (label, conf) = skill.guessWithConfidence(doubleArrayOf(x, y))
            askResult.text = "$label (${pct(conf)})"
            plog("You asked about ($x, $y) → it says $label, ${pct(conf)} sure.")
        }
        pane.addView(row(askX, askY, askBtn, askResult))

        val saveBtn = Button(this).apply { text = "Save" }
        val loadBtn = Button(this).apply { text = "Load" }
        val resetBtn = Button(this).apply { text = "Start over" }
        saveBtn.setOnClickListener {
            skillFile().writeText(skill.save())
            plog("Saved its brain (${skill.net.weightCount()} weights).")
        }
        loadBtn.setOnClickListener {
            if (!skillFile().exists()) { plog("Nothing saved yet."); return@setOnClickListener }
            registry.remove(skill.name)
            skill = Skill.load(skillFile().readText())
            plog("Loaded its saved brain.")
        }
        resetBtn.setOnClickListener {
            registry.remove(skill.name)
            skill = newCircleSkill()
            plog("Fresh start — it forgot everything. Teach it again!")
        }
        pane.addView(row(saveBtn, loadBtn, resetBtn))

        val (scroll, log) = makeLog("Ready. Tap “▶ See it learn”, or teach it yourself above.\n")
        patternScroll = scroll; patternLog = log
        pane.addView(scroll)

        return pane
    }

    // ---------- Code pane ----------

    private fun buildCodePane(): LinearLayout {
        val pane = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
        }

        pane.addView(card(
            "It learns to write tiny programs — then actually runs them.\n\n" +
            "Each round it writes new programs, runs each one, keeps the ones that " +
            "work, and studies them. Watch the share of working programs climb."
        ))

        pane.addView(section("Its little language"))
        val guide = TextView(this).apply {
            text = "x = 3 + 4        give x the value 7\n" +
                   "print(x * 2)     show 14\n" +
                   "if x > 2:        if true, do the indented lines\n" +
                   "    y = 99\n" +
                   "while x > 0:     repeat while true\n" +
                   "    x = x - 1"
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setBackgroundColor(0xFFF5F5F5.toInt())
            setPadding(20, 16, 20, 16)
        }
        pane.addView(guide)

        val demoBtn = Button(this).apply { text = "▶  See it learn to code" }
        demoBtn.setOnClickListener { runInBackground(demoBtn) { runCodeDemo() } }
        pane.addView(demoBtn)

        val trainBtn = Button(this).apply { text = "Study examples" }
        val selfBtn = Button(this).apply { text = "Self-teach loop" }
        trainBtn.setOnClickListener {
            runInBackground(trainBtn, selfBtn) {
                clog("Studying ${CodeCorpus.programs.size} example programs... (takes about a minute)")
                val rnn = CharRNN(CodeCorpus.vocab, hiddenSize = 128, learningRate = 0.05, seed = 7L)
                rnn.train(CodeCorpus.programs, epochs = 150)
                codeRnn = rnn
                clog("Done studying (${rnn.weightCount()} weights). Now run the self-teach loop!")
            }
        }
        selfBtn.setOnClickListener {
            runInBackground(trainBtn, selfBtn) {
                val rnn = codeRnn ?: run { clog("Tap “Study examples” first."); return@runInBackground }
                clog("Self-teaching: writing programs, running them, keeping the ones that work...")
                val corpus = CodeCorpus.programs.toMutableList()
                val reports = CodeSelfTrainer(24, 5, 30, 0.8).selfTeach(
                    rnn, corpus, codeInterp, System.currentTimeMillis())
                for (r in reports)
                    clog("  round ${r.round}: ${"%.0f".format(r.validPct * 100)}% of its programs ran — kept ${r.kept}")
                clog("Done — its library grew to ${corpus.size} programs, most written by itself.")
            }
        }
        pane.addView(row(trainBtn, selfBtn))

        val writeBtn = Button(this).apply { text = "Write a program" }
        val runBtn = Button(this).apply { text = "Run it" }
        writeBtn.setOnClickListener {
            val rnn = codeRnn ?: run { clog("Tap “Study examples” first."); return@setOnClickListener }
            val rng = Random(System.currentTimeMillis())
            val seedLine = CodeCorpus.programs.random(rng).lineSequence().first()
            lastProgram = rnn.sample(seedLine + "\n", 140, 0.8, rng).trim() + "\n"
            clog("— it wrote this: —\n$lastProgram")
        }
        runBtn.setOnClickListener {
            if (lastProgram.isEmpty()) { clog("Tap “Write a program” first."); return@setOnClickListener }
            when (val res = codeInterp.run(lastProgram)) {
                is TinyPy.Result.Ok -> clog("It runs! Output: ${res.output}")
                is TinyPy.Result.Err -> clog("It crashed: ${res.message}")
            }
        }
        pane.addView(row(writeBtn, runBtn))

        val (scroll, log) = makeLog("Welcome to the Code Lab! Tap “▶ See it learn to code” for the full show.\n")
        codeScroll = scroll; codeLog = log
        pane.addView(scroll)

        return pane
    }

    // ---------- scripted demos ----------

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
        val validation = (1..150).map { val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)); Example(p, circleLabel(p)) }
        val before = skill.accuracy(validation)
        val reports = skill.selfTrain(unlabeled, validation, SelfTrainer(0.85, 6, 200, 3))
        for (r in reports) plog("  round ${r.round}: ${pct(r.valAccuracy)} — it labeled ${r.pseudoLabeledAdded} points itself")
        plog("Finished: ${pct(before)} → ${pct(skill.accuracy(validation))}. Now quiz it yourself above!")
    }

    private fun runCodeDemo() {
        clog("Step 1: studying ${CodeCorpus.programs.size} example programs... (about a minute)")
        val rnn = CharRNN(CodeCorpus.vocab, hiddenSize = 128, learningRate = 0.05, seed = 7L)
        rnn.train(CodeCorpus.programs, epochs = 150)
        codeRnn = rnn
        clog("Step 2: now it teaches itself — writing, running, keeping what works.")
        val corpus = CodeCorpus.programs.toMutableList()
        val reports = CodeSelfTrainer(24, 4, 30, 0.8).selfTeach(rnn, corpus, codeInterp, System.currentTimeMillis())
        for (r in reports)
            clog("  round ${r.round}: ${"%.0f".format(r.validPct * 100)}% ran — kept ${r.kept} new programs")
        val rng = Random(System.currentTimeMillis())
        val seedLine = CodeCorpus.programs.random(rng).lineSequence().first()
        lastProgram = rnn.sample(seedLine + "\n", 140, 0.8, rng).trim() + "\n"
        clog("Step 3: here's one it wrote all by itself:\n$lastProgram")
        when (val res = codeInterp.run(lastProgram)) {
            is TinyPy.Result.Ok -> clog("And it runs! Output: ${res.output}")
            is TinyPy.Result.Err -> clog("This one crashed (${res.message}) — try “Write a program” for another.")
        }
    }

    // ---------- helpers ----------

    private fun circleTestAccuracy(n: Int): Double {
        val rng = Random(99L)
        val test = (1..n).map {
            val p = doubleArrayOf(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0))
            Example(p, circleLabel(p))
        }
        return skill.accuracy(test)
    }

    private fun skillFile() = java.io.File(filesDir, "skill.txt")

    private fun pct(d: Double) = "${"%.0f".format(d * 100)}%"

    private fun title(t: String, size: Float) = TextView(this).apply {
        text = t; textSize = size; setTypeface(typeface, Typeface.BOLD); setPadding(0, 0, 0, 2)
    }

    private fun subtitle(t: String) = TextView(this).apply {
        text = t; textSize = 14f; setTextColor(0xFF666666.toInt()); setPadding(0, 0, 0, 12)
    }

    private fun section(t: String) = TextView(this).apply {
        text = t; textSize = 16f; setTypeface(typeface, Typeface.BOLD); setPadding(0, 18, 0, 4)
    }

    private fun card(t: String) = TextView(this).apply {
        text = t; textSize = 14f; setBackgroundColor(0xFFF0F4FF.toInt()); setPadding(24, 18, 24, 18)
    }

    private fun hint(t: String) = TextView(this).apply {
        text = t; textSize = 12f; setTextColor(0xFF777777.toInt()); setPadding(0, 2, 0, 6)
    }

    private fun tabButton(t: String) = Button(this).apply {
        text = t; textSize = 16f
        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
    }

    private fun numInput(h: String) = android.widget.EditText(this).apply {
        hint = h
        inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED or InputType.TYPE_NUMBER_FLAG_DECIMAL
        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
    }

    private fun row(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        for (v in views) addView(v)
    }

    private fun makeLog(firstLine: String): Pair<ScrollView, TextView> {
        val tv = TextView(this).apply {
            text = firstLine; textSize = 13f; typeface = Typeface.MONOSPACE; setPadding(8, 8, 8, 8)
        }
        val sv = ScrollView(this).apply {
            addView(tv)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            setBackgroundColor(0xFFFAFAFA.toInt())
            setPadding(0, 8, 0, 0)
        }
        return sv to tv
    }

    private fun plog(msg: String) = logTo(patternLog, patternScroll, msg)
    private fun clog(msg: String) = logTo(codeLog, codeScroll, msg)

    private fun logTo(tv: TextView, sv: ScrollView, msg: String) {
        runOnUiThread {
            tv.append(msg + "\n")
            sv.post { sv.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    private fun runInBackground(vararg buttons: Button, work: () -> Unit) {
        buttons.forEach { it.isEnabled = false }
        Thread {
            try { work() } catch (e: Exception) { plog("Oops, something went wrong: ${e.message}") }
            finally { runOnUiThread { buttons.forEach { it.isEnabled = true } } }
        }.start()
    }
}
