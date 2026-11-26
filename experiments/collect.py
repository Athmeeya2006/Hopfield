"""Merge the outputs of the per-experiment Kaggle kernels (mounted under
/kaggle/input) into results/ and figures/, re-run the merged E1 analysis
and the Holm-corrected primary tests."""
import glob
import os
import shutil
import subprocess
import sys

import common

found = sorted(glob.glob("/kaggle/input/**/out/status_*.txt", recursive=True))
print("job outputs:", found)
for st in found:
    out = os.path.dirname(st)
    for sub, dst in (("results", common.RESULTS), ("figures", common.FIGURES),
                     ("demo/out", os.path.join(common.ROOT, "demo", "out")), ("logs", os.path.join(common.ROOT, "logs"))):
        src = os.path.join(out, sub)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
# re-run every analysis + figure on the merged raw data (cheap; uses the final analysis code)
CMDS = [["partA.py", "--analyze-only"], ["e1_capacity.py", "--analyze-only"],
        ["e2_generalization.py", "--analyze-only"], ["e3_regimes.py", "--analyze-only"],
        ["e4_dynamics.py", "--analyze-only"], ["e5_lesions.py", "--analyze-only"],
        ["e6_attention.py", "--analyze-only"], ["e7_forgetting.py", "--analyze-only"],
        ["extensions.py", "all", "--plot-only"], ["hypothesis_tests.py"]]
for cmd in CMDS:
    r = subprocess.run([sys.executable] + cmd, cwd=here)
    print(cmd, "->", r.returncode, flush=True)
    if r.returncode and cmd[0] in ("e1_capacity.py", "e2_generalization.py", "hypothesis_tests.py"):
        sys.exit(r.returncode)
