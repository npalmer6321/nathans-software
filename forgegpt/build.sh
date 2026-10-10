#!/bin/bash
# Build ForgeGPT into a runnable fat JAR (includes the Kotlin runtime).
# Linux/macOS. On Windows just run: java -jar forgegpt.jar (no build needed).
set -e
cd "$(dirname "$0")"

KOTLINC="${KOTLINC:-$HOME/workspace/toolchain/kotlinc/bin/kotlinc}"
if [ ! -x "$KOTLINC" ]; then
  echo "kotlinc not found at $KOTLINC"
  echo "Set KOTLINC=/path/to/kotlinc and retry."
  exit 1
fi

if [ -z "$JAVA_HOME" ]; then
  if [ -d "$HOME/workspace/toolchain/jvm/jdk-21.0.12.1+1-jre" ]; then
    export JAVA_HOME="$HOME/workspace/toolchain/jvm/jdk-21.0.12.1+1-jre"
  fi
fi
export PATH="$JAVA_HOME/bin:$PATH"

echo "Compiling ForgeGPT..."
# Only one `fun main` exists (Main.kt); the GUI is launched from it via the
# `gui` command, so the JAR manifest keeps Main-Class: forgegpt.MainKt.
"$KOTLINC" src -include-runtime -d forgegpt.jar
echo "Built: $(pwd)/forgegpt.jar"
ls -la forgegpt.jar
