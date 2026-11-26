"""E2: memorization to generalization (H2, main experiment).

P memories are stored around K unstored prototypes with similarity c in
{0.2, 0.4, 0.6, 0.8}; the number of memories per prototype p grows.  We
present corrupted cues of stored memories and corrupted prototypes (the DRM
lure), classify every final state as target / other stored memory /
prototype / spurious, and find the load P* (memories per prototype) at which
prototype recall overtakes target recall.
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.metrics import CATEGORIES, classify
from hopfield.models import Classical, ContinuousExp, Dense
from hopfield.patterns import flip_bits, prototype_patterns, take_per_prototype
from hopfield.seeds import int_seed, rng
from hopfield.stats import N_BOOT, boot_p_value

N = 100
K = 3
ETA = 0.1
CS = [0.2, 0.4, 0.6, 0.8]
PGRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
MODELS = ["M1-Hebb", "M2-n3", "M2-n4", "M2-exp", "M3-b0.1", "M3-b0.2", "M3-b1"]
if QUICK:
    PGRID = [1, 4, 16, 64]


def make_model(name, xi):
    if name == "M1-Hebb":
        return Classical.hebbian(xi)
    if name.startswith("M2"):
        return Dense(xi, {"M2-n3": 3, "M2-n4": 4, "M2-exp": "exp"}[name])
    return ContinuousExp(xi, float(name.split("-b")[1]))


def run_cues(model, xi, labels, protos, targets, mem_cues, lure_k, lure_cues, dyn_key):
    """Returns (mem_cat, mem_time, lure_cat, lure_time, mem_mproto)."""
    if isinstance(model, ContinuousExp):
        Xm, tm, _ = model.recall_batch(mem_cues)
        Xl, tl, _ = model.recall_batch(lure_cues)
        mem_states, lure_states = Xm, Xl
    else:
        mem_states, tm, lure_states, tl = [], [], [], []
        for j in range(len(mem_cues)):
            r = model.recall(mem_cues[j], seed=int_seed(*dyn_key, "m", j))
            mem_states.append(r.state)
            tm.append(r.time)
        for j in range(len(lure_cues)):
            r = model.recall(lure_cues[j], seed=int_seed(*dyn_key, "l", j))
            lure_states.append(r.state)
            tl.append(r.time)
    mc = [classify(mem_states[j], t, xi, protos[labels[t]]) for j, t in enumerate(targets)]
    lc = [classify(lure_states[j], None, xi, protos[k]) for j, k in enumerate(lure_k)]
    return (np.array([c[0] for c in mc]), np.asarray(tm, float),
            np.array([c[0] for c in lc]), np.asarray(tl, float),
            np.array([c[3] for c in mc]))


def task(args):
    name, c, s = args
    protos, mems = prototype_patterns(rng("E2", s, int(round(c * 100))), K, max(PGRID), N, c)
    out = dict(mem=np.zeros((len(PGRID), 4)), lure=np.zeros((len(PGRID), 4)),
               t_mem=np.full((len(PGRID), 4), np.nan), t_lure=np.full((len(PGRID), 4), np.nan),
               mproto=np.zeros(len(PGRID)))
    for ip, p in enumerate(PGRID):
        xi, labels = take_per_prototype(mems, p)
        cr = rng("E2-cue", s, int(round(c * 100)), p)
        targets = cr.choice(len(xi), size=N_CUES, replace=len(xi) < N_CUES)
        mem_cues = np.stack([flip_bits(cr, xi[t], ETA) for t in targets])
        lure_k = cr.integers(0, K, size=N_CUES)
        lure_cues = np.stack([flip_bits(cr, protos[k], ETA) for k in lure_k])
        model = make_model(name, xi)
        mcat, tm, lcat, tl, mp = run_cues(model, xi, labels, protos, targets, mem_cues,
                                          lure_k, lure_cues, ("E2-dyn", s, int(c * 100), p))
        for k in range(4):
            out["mem"][ip, k] = np.mean(mcat == k)
            out["lure"][ip, k] = np.mean(lcat == k)
            if np.any(mcat == k):
                out["t_mem"][ip, k] = tm[mcat == k].mean()
            if np.any(lcat == k):
                out["t_lure"][ip, k] = tl[lcat == k].mean()
        out["mproto"][ip] = mp.mean()
    return out


def run():
    warmup()
    tasks = [(m, c, s) for m in MODELS for c in CS for s in range(N_SETS)]
    with Timer("E2"):
        outs = pmap(task, tasks)
    data = {}
    for (m, c, s), o in zip(tasks, outs):
        dd = data.setdefault(m, {}).setdefault(str(c), {})
        for k, v in o.items():
            dd.setdefault(k, []).append(v)
    save_json("e2_raw.json", dict(N=N, K=K, eta=ETA, cs=CS, pgrid=PGRID, models=MODELS,
                                  categories=CATEGORIES, data=data))
    analyze()


def _cross(diff):
    """Load at which prototype recall overtakes target recall.

    diff = target - prototype fraction on the p grid.  The crossover is the
    first grid point i with diff < 0 (prototype strictly ahead); the
    crossing is interpolated (log p) between i and the last earlier point j
    with diff > 0.  Points where both are equal (e.g. both 0 because recall
    collapsed to spurious states) do not count as a crossover.  Returns +inf
    if the prototype never overtakes within the grid, -inf if it is ahead
    from the first point.
    """
    x = np.log(np.asarray(PGRID_A, float))
    neg = np.where(np.asarray(diff) < 0)[0]
    if neg.size == 0:
        return np.inf
    i = neg[0]
    pos = np.where(np.asarray(diff[:i]) > 0)[0]
    if pos.size == 0:
        return -np.inf
    j = pos[-1]
    d0, d1 = diff[j], diff[i]
    f = d0 / (d0 - d1)
    return float(np.exp(x[j] + f * (x[i] - x[j])))


PGRID_A = PGRID


def analyze():
    global PGRID_A
    d = load_json("e2_raw.json")
    PGRID_A = d["pgrid"]
    brng = np.random.default_rng(11)
    idx = brng.integers(0, N_SETS, size=(N_BOOT, N_SETS))
    summ = {}
    boots = {}
    for m in d["models"]:
        for c in d["cs"]:
            mem = np.array(d["data"][m][str(c)]["mem"])
            lure = np.array(d["data"][m][str(c)]["lure"])
            est = _cross(mem.mean(0)[:, 0] - mem.mean(0)[:, 2])
            b = np.array([_cross(mem[ix].mean(0)[:, 0] - mem[ix].mean(0)[:, 2]) for ix in idx])
            boots[(m, c)] = b
            lure_proto = lure[:, :, 2].mean(0)
            summ.setdefault(m, {})[str(c)] = dict(
                Pstar=est,
                lo=np.percentile(b, 2.5, method="lower"),
                hi=np.percentile(b, 97.5, method="higher"),
                frac_boot_censored=float(np.mean(~np.isfinite(b))),
                lure_false_recall=lure_proto,
                lure_P50=_cross(0.5 - lure_proto),
            )
    # H2 primary: P*(M2 n=3) > P*(M1) at c = 0.4  (paired bootstrap of log ratio)
    a, bb = boots[("M2-n3", 0.4)], boots[("M1-Hebb", 0.4)]
    lr = np.log(a) - np.log(bb)
    est = np.log(summ["M2-n3"]["0.4"]["Pstar"]) - np.log(summ["M1-Hebb"]["0.4"]["Pstar"])
    primary = dict(test="log P*(M2, n=3) - log P*(M1 Hebbian) at c = 0.4 (paired bootstrap over sets)",
                   estimate=est, ratio=float(np.exp(est)), lo=np.nanpercentile(lr, 2.5, method="lower"),
                   hi=np.nanpercentile(lr, 97.5, method="higher"), p=boot_p_value(lr, 0.0))
    # secondary: ordering across all models / c
    save_json("e2_summary.json", dict(models=summ, primary_H2=primary))
    for m in summ:
        print(m, {c: (round(v["Pstar"], 1) if np.isfinite(v["Pstar"]) else v["Pstar"]) for c, v in summ[m].items()})
    print("H2 primary:", primary)


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import INK, STATE_COLORS, save, style
    style()
    d = load_json("e2_raw.json")
    s = load_json("e2_summary.json")
    pg = np.array(d["pgrid"])
    models = d["models"]
    cats = d["categories"]
    fig, axes = plt.subplots(len(models), len(d["cs"]), figsize=(13, 2.1 * len(models)),
                             sharex=True, sharey=True)
    for i, m in enumerate(models):
        for j, c in enumerate(d["cs"]):
            ax = axes[i, j]
            mem = np.array(d["data"][m][str(c)]["mem"])
            mu = mem.mean(0)
            lo, hi = np.percentile(mem, [2.5, 97.5], axis=0)  # set-level spread
            for k, cat in enumerate(cats):
                ax.plot(pg, mu[:, k], "o-", ms=2.5, color=STATE_COLORS[cat], label=cat)
            lure = np.array(d["data"][m][str(c)]["lure"])[:, :, 2].mean(0)
            ax.plot(pg, lure, "--", color=STATE_COLORS["prototype"], lw=1, label="lure -> prototype")
            ps = s["models"][m][str(c)]["Pstar"]
            if isinstance(ps, (int, float)) and np.isfinite(ps):
                ax.axvline(ps, color=INK, lw=0.8, ls=":")
                ax.text(ps, 1.02, f"P*={ps:.0f}", fontsize=6, ha="center")
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.1)
            if i == 0:
                ax.set_title(f"c = {c}\n", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{m}\nfraction", fontsize=8)
            if i == len(models) - 1:
                ax.set_xlabel("memories per prototype p")
    axes[0, -1].legend(fontsize=6, loc="center right")
    fig.suptitle(f"E2: final-state fractions vs load (N = {d['N']}, K = {d['K']}, cue eta = {d['eta']})", y=1.0)
    save(fig, f"{FIGURES}/e2_state_fractions.png")

    # P* summary
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    cols = plt.cm.tab10(np.arange(len(models)))
    for m, col in zip(models, cols):
        ys, los, his, cs = [], [], [], []
        for c in d["cs"]:
            v = s["models"][m][str(c)]
            p = v["Pstar"]
            if isinstance(p, str) or p is None:
                continue
            cs.append(c)
            ys.append(p)
            los.append(p - v["lo"] if isinstance(v["lo"], (int, float)) else 0)
            his.append(v["hi"] - p if isinstance(v["hi"], (int, float)) else 0)
        if cs:
            ax.errorbar(cs, ys, yerr=[np.maximum(los, 0), np.maximum(his, 0)], fmt="o-", color=col,
                        label=m, capsize=2)
    ax.set_yscale("log")
    ax.set_xlabel("similarity c")
    ax.set_ylabel("P* (memories per prototype)")
    ax.set_title("Crossover load P* (models with no crossover omitted)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for m, col in zip(models, cols):
        lure = np.array(d["data"][m]["0.4"]["lure"])[:, :, 2]
        ax.plot(pg, lure.mean(0), "o-", ms=3, color=col, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("number of studied associates per lure (p)")
    ax.set_ylabel("false recall of the lure (prototype)")
    ax.set_title("DRM analogue, c = 0.4: false recall vs associates")
    ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/e2_pstar_drm.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.analyze_only:
        analyze()
    elif not a.plot_only:
        run()
    plot()
