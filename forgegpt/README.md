# ForgeGPT

A **from-scratch transformer language model** — the same architecture family as
real LLMs like GPT — written in pure Kotlin with **zero ML frameworks**.
Every matrix multiply, softmax, layer norm, attention head, backprop step and
Adam update is hand-written. No PyTorch, no TensorFlow, no ONNX, no NumPy.

It learns next-token prediction on public-domain English books and can
generate text continuations from a prompt.

## What's inside

```
forgegpt/
  forgegpt.jar          <- prebuilt runnable JAR (just needs Java 17+)
  model.fg              <- trained demo weights (shipped, ready to generate)
  vocab.txt             <- vocabulary used by the demo model
  corpus/corpus.txt     <- training corpus (public-domain books, cleaned)
  src/forgegpt/
    Mat.kt              <- hand-written float linear algebra (matmul, softmax, layer norm)
    Tokenizer.kt        <- word-level tokenizer, vocab built from corpus
    Model.kt            <- decoder-only transformer + full manual backprop
    Adam.kt             <- Adam optimizer, hand-written
    Train.kt            <- mini-batch training loop (multi-threaded)
    Generate.kt         <- causal sampling (temperature + top-k)
    GradCheck.kt        <- numerical gradient self-test
    Main.kt             <- CLI (train / generate / gradcheck)
  .vscode/
    tasks.json          <- one-click "Train model" / "Generate text" tasks
    launch.json         <- F5 run configs (needs Extension Pack for Java)
  build.sh              <- rebuild the JAR from source (Linux/macOS)
  train.log             <- log from the shipped proof training run
```

## Architecture

Decoder-only transformer (GPT family), pre-norm blocks:

```
tokens -> token embedding + learned positional embedding
       ->  L x [ x += CausalMultiHeadAttn(LayerNorm(x))
               ; x += MLP(ReLU, LayerNorm(x)) ]
       -> LayerNorm -> linear -> vocabulary logits
```

Default (shipped) configuration:

| Hyperparameter | Value |
|---|---|
| Embedding width (d_model) | 64 |
| Transformer blocks | 3 |
| Attention heads | 4 (16 dims each, causal masking) |
| Context length | 48 tokens |
| MLP hidden size | 256 (ReLU) |
| Vocabulary | 2,500 word-level tokens |
| Parameters | **475,652** |
| Optimizer | Adam, lr 3e-4, grad-norm clip 1.0 |
| Loss | next-token cross-entropy |

The backprop is verified two ways: a built-in numerical gradient check
(`gradcheck` command, passes on 0/1/2-block configs) and an independent
float64 numpy reimplementation that matches every activation and gradient to
~1e-6.

## Run on Windows (Nathan's PC)

You do **not** need Kotlin or any build tools — only Java.

**1. Install Java 17+** — pick one:
- Easiest: open PowerShell and run `winget install Microsoft.OpenJDK.17`
- Or download the installer: https://aka.ms/download-jdk/microsoft-jdk-17-windows-x64.msi
  (all downloads: https://learn.microsoft.com/en-us/java/openjdk/download)

Verify: open a terminal and run `java -version` — you should see 17 or higher.

**2. Unzip** `forgegpt-windows.zip` anywhere, e.g. `C:\forgegpt`.

**3. Open the folder in VS Code** (File -> Open Folder... -> select it).

**4. Run it — three ways:**
- **Tasks (no extensions needed):** `Terminal -> Run Task...` -> **"ForgeGPT: Train model"**
  or **"ForgeGPT: Generate text"** (it will ask for a prompt).
- **F5:** install the *"Extension Pack for Java"* in VS Code, then press **F5**
  and pick **"ForgeGPT: Train"** or **"ForgeGPT: Generate"**.
- **Terminal fallback** (works anywhere, VS Code not required):

```powershell
cd C:\forgegpt
# Generate text with the shipped trained model:
java -jar forgegpt.jar generate --model model.fg --vocab vocab.txt --prompt "once upon a time" --max-tokens 80 --temperature 0.8
# Train your own (takes a while on CPU; start small):
java -jar forgegpt.jar train --corpus corpus/corpus.txt --out mymodel.fg --vocab-out myvocab.txt --steps 600
```

## Build from source (Linux/macOS)

```bash
./build.sh            # needs kotlinc + a JDK; produces forgegpt.jar
java -jar forgegpt.jar train    --corpus corpus/corpus.txt --steps 600
java -jar forgegpt.jar generate --model model.fg --vocab vocab.txt \
    --prompt "the king said" --max-tokens 60 --temperature 0.8 --top-k 40
java -jar forgegpt.jar gradcheck
```

Training options: `--steps N --batch N --lr F --d-model N --layers N --heads N
--seq-len N --ffn N --vocab N --seed N --log-every N --sample-every N
--checkpoint-every N`. Generation options: `--prompt "..." --max-tokens N
--temperature F --top-k N --seed N` (`--temperature 0` = greedy).

## Proof run (shipped model)

Trained on CPU in two phases (phase 2 resumed from phase-1 weights with
`--init-model`): 8,000 steps x (8 sequences x 48 tokens) = ~3.07M tokens,
about 3.7 passes over the corpus, ~29 minutes on 2 CPU cores.

- Corpus: 834,399 tokens from 6 public-domain books
  (Alice's Adventures in Wonderland, Pride and Prejudice, The Adventures of
  Sherlock Holmes, Frankenstein, A Tale of Two Cities, Moby Dick — Project
  Gutenberg, headers/footers stripped, ASCII-normalized;
  reproducible via `corpus/fetch_and_clean.py`)
- Parameters: 475,652 | Vocab: 2,500 | Threads: 2
- Loss: **7.87 (random guessing ≈ 7.82) -> 4.07** — clearly learning,
  plateaued by the end
- Full log: `train.log` (both phases, appended)

Sample generations from the shipped `model.fg`
(`--temperature 0.55 --top-k 25`):

```
> "alice looked"
alice looked.""that _ _ _ _ _,"said alice,"and do not do not say,"said
alice."i know you will not you,"said mr. bingley.""i am not at.""she is.""i am

> "it was a dark"
it was a dark man. he was not a - -""i am not a for a matter."".""no,
yes,"said alice, i know that you must have been a little.""said the
hatter,"i don't see what i do not
```

Recognizably English, and it even picked up corpus-specific voices
("said alice", "said mr. bingley", "said the hatter") — but it rambles,
repeats dialogue tics (`""`), and can't hold a coherent thought. Exactly
what a half-million parameters buys you; see Honest limits.

## How it works (30-second version)

1. **Tokenizer** splits text into lowercase words/punctuation and maps the
   2,500 most frequent to ids (`<pad> <unk> <bos> <eos>` reserved).
2. **Training** feeds random 49-token windows; the model predicts each next
   token; cross-entropy loss is backpropagated through all 3 transformer
   blocks by hand-written calculus; Adam updates the weights. Batches run on
   a small thread pool.
3. **Generation** runs the prompt through the model, samples the next token
   from the last position's logits (temperature + top-k), appends it, repeats.

## Honest limits

- This is a **~0.5M-parameter** model trained on **~3.1M tokens** (~29 minutes on 2 CPU cores).
  Real LLMs are **billions–trillions** of parameters trained on **trillions**
  of tokens for **weeks on thousands of datacenter GPUs**. No PC build can
  reach that — the gap is ~6 orders of magnitude of compute, not a smarter
  algorithm away.
- What it actually learns: **surface patterns of English** — common words,
  local grammar, frequent phrases, some style of the training books. It does
  not "understand", reason, or remember facts; long generations wander and
  contradict themselves.
- Word-level vocab (2,500) means rare words become `<unk>`; context is only
  48 tokens, so it can't track anything far back.
- CPU training is slow: roughly 2–3 steps/sec on 2 cores at this size.
  Bigger configs are possible (`--d-model 128 --layers 4 ...`) but need
  proportionally more patience.

## License note

Code is original. The corpus is public-domain literature from Project
Gutenberg (US). The trained weights inevitably memorize short fragments of
those books — don't republish long generations as your own writing.
