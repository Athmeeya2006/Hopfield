"""Collect the pre-specified primary test of every hypothesis and apply the
Holm correction across H1-H6 (proposal section 5).  All other comparisons in
the experiment summaries are exploratory."""
from __future__ import annotations

import common  # noqa: F401
import numpy as np

from common import load_json, save_json
from hopfield.stats import holm

SOURCES = {"H1": ("e1_summary.json", "primary_H1"), "H2": ("e2_summary.json", "primary_H2"),
           "H3": ("e3_summary.json", "primary_H3"), "H4": ("e5_summary.json", "primary_H4"),
           "H5": ("e6_summary.json", "primary_H5"), "H6": ("e7_summary.json", "primary_H6")}
# predicted direction of each primary statistic
DIRECTION = {
    "H1": lambda t: t["estimate"] > 0 and t["lo"] > 0,
    "H2": lambda t: t["estimate"] > 0 and t["lo"] > 0,
    "H3": lambda t: t["median_rho"] > 0,
    "H4": lambda t: t["mean_M4"] > t["mean_M1"],
    "H5": lambda t: t["mean_rho0"] > t["mean_rho08"],
    "H6": lambda t: t["mean_bounded"] > t["mean_unbounded"],
}


def main():
    tests = {}
    for h, (f, k) in SOURCES.items():
        try:
            tests[h] = load_json(f)[k]
        except (FileNotFoundError, KeyError) as e:
            print(f"{h}: missing ({e})")
    pv = {h: t["p"] for h, t in tests.items() if t.get("p") is not None and np.isfinite(t["p"])}
    adj = holm(pv)
    for h, t in tests.items():
        t["p_holm"] = adj.get(h)
        try:
            t["direction_as_predicted"] = bool(DIRECTION[h](t))
        except TypeError:  # missing (None/NaN) statistic
            t["direction_as_predicted"] = False
        t["supported"] = bool(t["direction_as_predicted"] and t["p_holm"] is not None and t["p_holm"] < 0.05)
        print(f"{h}: p = {t['p']}, Holm p = {t['p_holm']}, direction ok = {t['direction_as_predicted']}, "
              f"supported = {t['supported']}")
    save_json("primary_tests.json", tests)


if __name__ == "__main__":
    main()
