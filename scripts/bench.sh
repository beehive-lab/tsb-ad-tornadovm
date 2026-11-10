#!/bin/bash
# Validates and benchmarks the TornadoVM detectors and VUS metrics against TSB-AD's CPU implementations.
#
# Needs TORNADOVM_HOME and JAVA_HOME set and PY pointing at a Python with the TSB-AD package installed.
#
#   scripts/bench.sh DATASET_DIR FILE_LIST OUTDIR
#
# Writes OUTDIR/ref (CPU scores, times and TSB-AD's VUS), OUTDIR/tvm (TornadoVM scores, times and VUS)
# and OUTDIR/summary.csv with OUTDIR/summary_per_file.csv. VUS_EVERY=n runs TSB-AD's slow VUS on every
# n-th file only (default 5); REPEAT sets the TornadoVM runs per detector (default 3, the first one
# includes JIT compilation).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA=${1:?usage: bench.sh DATASET_DIR FILE_LIST OUTDIR}
LIST=${2:?usage: bench.sh DATASET_DIR FILE_LIST OUTDIR}
OUT=${3:?usage: bench.sh DATASET_DIR FILE_LIST OUTDIR}
PY=${PY:-python3}
mkdir -p "$OUT"
(cd "$HERE" && ./mvnw -q -B compile)
$PY "$HERE/python/run_reference.py" --dataset_dir "$DATA" --file_list "$LIST" --out_dir "$OUT/ref" --vus_every "${VUS_EVERY:-5}" | tee "$OUT/ref.log"
java @"$TORNADOVM_HOME/tornado-argfile" -Dtornado.device.memory=8GB -cp "$HERE/target/classes" org.beehive.tsbad.TsbadBench \
  --dataset-dir "$DATA" --file-list "$LIST" --out-dir "$OUT/tvm" --ref-dir "$OUT/ref" --repeat "${REPEAT:-3}" 2>&1 | grep '\[tsbad\]' | tee "$OUT/tvm.log"
$PY "$HERE/python/compare.py" --ref "$OUT/ref" --tvm "$OUT/tvm" --summary "$OUT/summary.csv" | tee "$OUT/summary.txt"
