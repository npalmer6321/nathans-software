# SelfForge — a neural network that trains itself, on your phone

**v1.2:** brand-new friendly UI (two tabs: Patterns + Code, plain-language
onboarding, guided demos) and a bigger TinyPy language with `if` and `while`.

## What this is

A small neural network written **from scratch in pure Kotlin** — no ML frameworks,
no servers, no cloud. It lives entirely on the Android device and it learns
two ways:

1. **Taught** — you give it labeled examples ("this point is inside the circle").
2. **Self-trained (recursive)** — the real thing: the model labels unlabeled
   data *itself*, keeps only predictions it is confident **and** consistent
   about (same label two rounds in a row), adds them to its training set as
   if they were ground truth, and retrains. Each round's improved model
   teaches the next round — with zero new human labels.

**New skills/abilities:** every skill is its own network plus its examples.
Teaching the app a new ability = registering a new skill and giving it
examples. Skills save to a file (weights + examples) and reload bit-identical.

**Code lab:** a second, bigger brain — a character-level RNN (26k weights)
that learns to write programs in TinyPy (a tiny Python-like language with
assignment, `print`, `if`, `while`, and comparisons), then **runs them with
its own built-in interpreter**. Its recursive loop is generate → run → keep
the working ones → retrain, and you watch the share of runnable programs
climb each round.

**Guardrails** (so it can't silently get worse):
- *Consistency filter* — flip-flopping boundary guesses are never trusted.
- *Best-snapshot restore* — the weights from the best validation round are
  kept and restored at the end. Self-training never finishes worse than it
  started on the held-out set.
- *Loop protection* — runaway `while` loops are cut off and count as failures.
- It only learns from **data you give it / it observes**. It does not rewrite
  its own code or architecture.

## Verified (not just claimed)

The core (`NeuralNet`, `SelfTrainer`, `Skill` — pure Kotlin, zero Android
dependencies) was compiled with kotlinc and run on a JVM. The demo output:

- 122-weight network, taught with **8 human examples** → 91.0% validation accuracy
- Recursive self-training: **4 rounds, 571 self-labeled examples**, accuracy held
- Second independent skill trained to 90.3% (multi-skill works)
- Save → load round-trip: predictions bit-identical
- Live predictions correct on probe points
- Code lab: runnable TinyPy programs climbed **3% → 83%** over 6 self-teaching
  rounds (training set grew 71 → 321 programs, all machine-written and verified
  by actually running them). Full run: `selfforge.codegen.CodeDemoMainKt` (a few minutes).

Run it yourself (needs kotlinc + a JDK/JRE 17+ on PATH):

```bash
kotlinc core/src/main/kotlin/selfforge/*.kt core/src/main/kotlin/selfforge/codegen/*.kt \
  -include-runtime -d /tmp/selfforge-demo.jar
java -cp /tmp/selfforge-demo.jar selfforge.DemoMainKt      # patterns demo (the numbers above)
java -cp /tmp/selfforge-demo.jar selfforge.codegen.CodeDemoMainKt  # code-lab demo
```

## The Android app

Two friendly tabs, plain language throughout, guided "▶ See it learn" demos:

- **Patterns tab** — "Can it learn to see an invisible circle?" Teach it
  inside/outside examples, quiz it, then let it teach itself from unlabeled
  points. Save / Load / Start over included.
- **Code tab** — the code lab described above, plus a language guide card
  showing the TinyPy constructs with examples. Buttons: *Study examples*,
  *Self-teach loop*, *Write a program*, *Run it*.

## Build the APK / install

**Easiest:** a signed debug APK is included in this folder — install
`SelfForge.apk` on the phone (you'll need to allow installs from this
source once). Needs Android 8.0+.

**From source:** open the `android/` folder in Android Studio, let it sync,
then Build → Build APK(s). The app is plain Android framework (no androidx
deps in `MainActivity`), and the sources are compile-verified against the
Android 34 SDK. GUI entry point: `com.selfforge.app.MainActivity`.

## PC version

`SelfForge-PC.jar` (included in this folder) is the same app for desktop —
a double-clickable Java program with the same two tabs (Patterns + Code)
and the same self-teaching brains. Needs **Java 17 or newer** installed;
then just double-click it, or run `java -jar SelfForge-PC.jar`. The desktop
build uses slightly fuller training settings since PCs are faster than phones.

**Rebuild it from source:** with kotlinc on PATH, run `pc/build-pc.sh`
(it compiles `core/` + `pc/` and writes a double-clickable
`SelfForge-PC.jar` next to this README). GUI entry point:
`selfforge.pc.DesktopAppKt` (`pc/src/main/kotlin/selfforge/pc/DesktopApp.kt`).

## Honest limits

- This is a **tiny** network (hundreds of weights), not a large language model.
  It learns narrow pattern skills (e.g. classifying 2D points), not language,
  reasoning, or "everything an AI assistant does" — that needs datacenter-scale
  models and can't fit on a phone.
- "Recursive self-training" is the genuine ML technique of *pseudo-labeling*:
  the model really does expand its own training set and change its own
  weights. It does **not** redesign its architecture or edit its own code.
- Demo training sizes are tuned for phones; the full demo takes a minute or
  two on a modern device. Smaller/faster runs are one constant away
  (`epochsPerRound`, `maxRounds` in `MainActivity`).

## Layout

```
selfforge/
  core/                  pure-Kotlin learning engine (no Android deps)
    src/main/kotlin/selfforge/
      NeuralNet.kt       MLP from scratch: forward, backprop, SGD, save/load weights
      SelfTrainer.kt     recursive pseudo-labeling loop + guardrails
      Skill.kt           a "skill" = one network + its examples; SkillRegistry
      DemoMain.kt        JVM verification demo (the numbers above)
      codegen/
        CharRNN.kt       char-level RNN from scratch (learns to write code)
        TinyPy.kt        tiny Python-like interpreter (runs the model's programs)
        CodeCorpus.kt    example programs it learns from (now with if/while)
        CodeSelfTrainer.kt  generate -> run -> keep working ones -> retrain loop
        CodeDemoMain.kt  JVM verification
  android/               Android Studio project (includes core sources)
    app/src/main/java/com/selfforge/app/MainActivity.kt
  pc/                    desktop source — Swing GUI, same two tabs
    src/main/kotlin/selfforge/pc/DesktopApp.kt
  README.md
```
