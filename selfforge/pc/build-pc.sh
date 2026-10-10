#!/usr/bin/env bash
# Build the SelfForge PC desktop app (Swing GUI) into a runnable jar.
#
# Output: ../SelfForge-PC.jar  (double-clickable, Main-Class set)
# Needs: kotlinc on PATH (or set KOTLINC=/path/to/kotlinc) and a JDK/JRE 17+.
set -euo pipefail
cd "$(dirname "$0")"

KOTLINC="${KOTLINC:-kotlinc}"
OUT_JAR="../SelfForge-PC.jar"

echo "Compiling core + PC GUI..."
"$KOTLINC" \
  ../core/src/main/kotlin/selfforge/*.kt \
  ../core/src/main/kotlin/selfforge/codegen/*.kt \
  src/main/kotlin/selfforge/pc/*.kt \
  -include-runtime -d "$OUT_JAR"

# Make the jar double-clickable: inject Main-Class into the manifest
# (kotlinc does not write one). Uses python3's zipfile.
python3 - "$OUT_JAR" <<'PYEOF'
import sys, zipfile, shutil, os
jar = sys.argv[1]
tmp = jar + ".tmp"
manifest = "Manifest-Version: 1.0\r\nMain-Class: selfforge.pc.DesktopAppKt\r\n\r\n"
zin = zipfile.ZipFile(jar, "r")
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    if item.filename.upper() == "META-INF/MANIFEST.MF":
        continue
    zout.writestr(item, zin.read(item.filename))
zout.writestr("META-INF/MANIFEST.MF", manifest)
zout.close(); zin.close()
shutil.move(tmp, jar)
print("manifest written")
PYEOF

echo "Built $OUT_JAR"
echo "Run it:  java -jar SelfForge-PC.jar"
