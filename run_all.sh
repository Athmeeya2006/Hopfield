#!/usr/bin/env bash
# Regenerate every result and figure from the master seed.
#   QUICK=1 ./run_all.sh     smoke test (4 sets x 5 cues)
#   WORKERS=4 ./run_all.sh   limit parallel worker processes
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p logs
$PY -m pytest -q tests | tee logs/tests.log
cd experiments
for s in partA e1_capacity e2_generalization e3_regimes e4_dynamics e5_lesions e6_attention e7_forgetting; do
  $PY $s.py 2>&1 | tee ../logs/$s.log
done
$PY extensions.py all 2>&1 | tee ../logs/extensions.log
$PY hypothesis_tests.py 2>&1 | tee ../logs/hypothesis_tests.log
