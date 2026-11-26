"""E4: recovery dynamics (H2, H3).

For every model we record the overlap m(t) = (1/N) sum_i xi_i s_i(t) with the
cued memory and with its prototype, and the energy E(t), after every sweep.
Varied: cue corruption eta, load p and similarity c.  We compare convergence
time, initial and final overlap, and the trajectories of true (target),
false (prototype) and failed (other memory / spurious) recalls.  Convergence
time is a reaction-time analogue: are false (prototype) recalls faster or
slower than true ones?
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np
from scipy import stats as sstats

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_npz, pmap, save_json, save_npz, warmup
from hopfield.metrics import classify
from hopfield.models import Classical, ContinuousExp, Dense
from hopfield.patterns import flip_bits, prototype_patterns, take_per_prototype
from hopfield.seeds import int_seed, rng

N = 100
K = 3
CS = [0.4, 0.6]
ETAS = [0.1, 0.2, 0.3]
PS = [4, 16, 64]
MODELS = ["M1-Hebb", "M2-n3", "M2-n4", "M2-exp", "M3-b0.1"]
L = 51  # trajectory length (sweep 0..50)
EX = 6  # example trajectories kept per kind for the first 3 sets
if QUICK:
    ETAS = [0.2]
    PS = [4, 64]


def make_model(name, xi):
    if name == "M1-Hebb":
        return Classical.hebbian(xi)
    if name.startswith("M2"):
        return Dense(xi, {"M2-n3": 3, "M2-n4": 4, "M2-exp": "exp"}[name])
    return ContinuousExp(xi, float(name.split("-b")[1]))


def _pad(a):
    a = np.asarray(a, float)
    out = np.empty((L,) + a.shape[1:])
    n = min(len(a), L)
    out[:n] = a[:n]
    out[n:] = a[n - 1]  # hold the fixed point
    return out


def task(args):
    mi, ci, ei, pi, s = args
    name, c, eta, p = MODELS[mi], CS[ci], ETAS[ei], PS[pi]
    protos, mems = prototype_patterns(rng("E4", s, int(c * 100), p), K, p, N, c)
    xi, labels = take_per_prototype(mems, p)
    model = make_model(name, xi)
    cr = rng("E4-cue", s, int(c * 100), p, int(eta * 100))
    targets = cr.choice(len(xi), size=N_CUES, replace=len(xi) < N_CUES)
    lure_k = cr.integers(0, K, size=N_CUES)
    rows = []          # per cue: kind(0 mem/1 lure), cat, time, m0_t, m0_p, mT_t, mT_p
    traj_sum = np.zeros((2, 4, L, 3))   # kind x category x t x (m_target, m_proto, E_norm)
    traj_cnt = np.zeros((2, 4))
    examples = []
    for kind in (0, 1):
        for j in range(N_CUES):
            if kind == 0:
                t = targets[j]
                src, proto = xi[t], protos[labels[t]]
            else:
                t = None
                src, proto = protos[lure_k[j]], protos[lure_k[j]]
            cue = flip_bits(cr, src, eta)
            ref_t = xi[t] if t is not None else proto
            r = model.recall(cue, seed=int_seed("E4-dyn", s, mi, ci, ei, pi, kind, j),
                             refs=np.stack([ref_t, proto]), record=True)
            cat = classify(r.state, t, xi, proto)[0]
            mt = _pad(r.m_traj)
            E = _pad(r.E_traj)
            En = (E - E[-1]) / (abs(E[0] - E[-1]) + 1e-12)  # 1 -> 0 normalised energy descent
            traj_sum[kind, cat] += np.column_stack([mt, En])
            traj_cnt[kind, cat] += 1
            rows.append([kind, cat, r.time, mt[0, 0], mt[0, 1], mt[-1, 0], mt[-1, 1]])
            if s < 3 and j < min(EX, N_CUES):
                examples.append(np.column_stack([mt, En]))
    return np.array(rows), traj_sum, traj_cnt, np.array(examples)


def run():
    warmup()
    tasks = [(mi, ci, ei, pi, s) for mi in range(len(MODELS)) for ci in range(len(CS))
             for ei in range(len(ETAS)) for pi in range(len(PS)) for s in range(N_SETS)]
    with Timer("E4"):
        outs = pmap(task, tasks, chunksize=2)
    shape = (len(MODELS), len(CS), len(ETAS), len(PS), N_SETS)
    rows = np.zeros(shape + (2 * N_CUES, 7))
    tsum = np.zeros(shape + (2, 4, L, 3))
    tcnt = np.zeros(shape + (2, 4))
    ex = np.zeros(shape[:4] + (3, 2 * min(EX, N_CUES), L, 3))
    for (mi, ci, ei, pi, s), (r, ts, tc, e) in zip(tasks, outs):
        rows[mi, ci, ei, pi, s] = r
        tsum[mi, ci, ei, pi, s] = ts
        tcnt[mi, ci, ei, pi, s] = tc
        if s < 3:
            ex[mi, ci, ei, pi, s] = e
    save_npz("e4_raw.npz", rows=rows, tsum=tsum, tcnt=tcnt, examples=ex,
             models=np.array(MODELS), cs=np.array(CS), etas=np.array(ETAS), ps=np.array(PS))
    analyze()


def analyze():
    d = load_npz("e4_raw.npz")
    rows, models = d["rows"], list(d["models"])
    out = {"rt": {}, "summary": {}}
    for mi, m in enumerate(models):
        for ci, c in enumerate(d["cs"]):
            for ei, eta in enumerate(d["etas"]):
                for pi, p in enumerate(d["ps"]):
                    R = rows[mi, ci, ei, pi]  # sets x cues x 7
                    key = f"{m}_c{c}_eta{eta}_p{p}"
                    mem = R[:, :, 0] == 0
                    lure = R[:, :, 0] == 1
                    cat = R[:, :, 1]
                    tm = R[:, :, 2]

                    def per_set(mask):
                        v = np.where(mask, tm, np.nan)
                        with np.errstate(all="ignore"):
                            return np.nanmean(v, axis=1)
                    true_rt = per_set(mem & (cat == 0))
                    false_mem_rt = per_set(mem & (cat == 2))
                    false_lure_rt = per_set(lure & (cat == 2))
                    fail_rt = per_set(mem & ((cat == 1) | (cat == 3)))
                    res = dict(n_true=int(np.sum(mem & (cat == 0))), n_false_mem=int(np.sum(mem & (cat == 2))),
                               n_false_lure=int(np.sum(lure & (cat == 2))),
                               n_fail=int(np.sum(mem & ((cat == 1) | (cat == 3)))),
                               rt_true=np.nanmean(true_rt), rt_false_mem=np.nanmean(false_mem_rt),
                               rt_false_lure=np.nanmean(false_lure_rt), rt_fail=np.nanmean(fail_rt),
                               m0_target=float(R[:, :, 3][mem].mean()),
                               mfinal_target=float(R[:, :, 5][mem].mean()))
                    # paired comparison across sets where both outcomes occur
                    for lab, other in (("false_mem", false_mem_rt), ("false_lure", false_lure_rt)):
                        ok = np.isfinite(true_rt) & np.isfinite(other)
                        if ok.sum() >= 6 and np.any(other[ok] != true_rt[ok]):
                            w = sstats.wilcoxon(other[ok], true_rt[ok])
                            res[f"paired_{lab}_minus_true"] = dict(n_sets=int(ok.sum()),
                                                                  mean_diff=float(np.mean(other[ok] - true_rt[ok])),
                                                                  p=float(w.pvalue))
                    out["rt"][key] = res
    save_json("e4_summary.json", out)
    for k, v in out["rt"].items():
        if "paired_false_mem_minus_true" in v or "paired_false_lure_minus_true" in v:
            print(k, {a: b for a, b in v.items() if a.startswith("paired") or a.startswith("rt")})


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import STATE_COLORS, save, style
    style()
    d = load_npz("e4_raw.npz")
    models = list(d["models"])
    cats = ["target", "other memory", "prototype", "spurious"]
    ci = list(d["cs"]).index(0.4) if 0.4 in d["cs"] else 0
    ei = list(d["etas"]).index(0.2) if 0.2 in d["etas"] else 0
    # mean m(t) per outcome category (memory cues), all loads pooled, c = 0.4, eta = 0.2
    fig, axes = plt.subplots(3, len(models), figsize=(3.2 * len(models), 8), sharex=True)
    for j, m in enumerate(models):
        ts = d["tsum"][j, ci, ei].sum(axis=(0, 1))  # (kind, cat, L, 3)
        tc = d["tcnt"][j, ci, ei].sum(axis=(0, 1))
        for row, (q, lab) in enumerate([(0, "m with cued memory"), (1, "m with prototype"),
                                        (2, "normalised E(t)")]):
            ax = axes[row, j]
            for k, cat in enumerate(cats):
                if tc[0, k] >= 5:
                    ax.plot(np.arange(10), ts[0, k, :10, q] / tc[0, k], "o-", ms=3,
                            color=STATE_COLORS[cat], label=f"{cat} (n={int(tc[0, k])})")
            if tc[1, 2] >= 5:
                ax.plot(np.arange(10), ts[1, 2, :10, q] / tc[1, 2], "s--", ms=3,
                        color=STATE_COLORS["prototype"], alpha=0.6, label=f"lure->prototype (n={int(tc[1, 2])})")
            if row == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(lab)
            if row == 2:
                ax.set_xlabel("sweep (M3: iteration)")
            if row == 0:
                ax.legend(fontsize=5.5)
    fig.suptitle("E4: mean trajectories by outcome (c = 0.4, eta = 0.2, loads pooled)")
    save(fig, f"{FIGURES}/e4_trajectories.png")

    # recall time by outcome, per model and load
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), sharey=False)
    for j, m in enumerate(models):
        ax = axes[j]
        R = d["rows"][j, ci, ei]  # p x sets x cues x 7
        for pi, p in enumerate(d["ps"]):
            for k, (lab, mask_fn, col) in enumerate([
                    ("true", lambda r: (r[..., 0] == 0) & (r[..., 1] == 0), STATE_COLORS["target"]),
                    ("false (mem cue)", lambda r: (r[..., 0] == 0) & (r[..., 1] == 2), STATE_COLORS["prototype"]),
                    ("false (lure)", lambda r: (r[..., 0] == 1) & (r[..., 1] == 2), "#e0b070"),
                    ("failed", lambda r: (r[..., 0] == 0) & ((r[..., 1] == 1) | (r[..., 1] == 3)), "#8a8a8a")]):
                mask = mask_fn(R[pi])
                if mask.sum() >= 5:
                    v = R[pi][..., 2][mask]
                    ax.errorbar(pi + (k - 1.5) * 0.18, v.mean(), yerr=1.96 * v.std() / np.sqrt(len(v)),
                                fmt="o", color=col, label=lab if pi == 0 else None, capsize=2)
        ax.set_xticks(range(len(d["ps"])))
        ax.set_xticklabels([f"p={p}" for p in d["ps"]])
        ax.set_title(m)
        ax.set_ylabel("recall time (sweeps; M3: iterations)")
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle("E4: recall time (reaction-time analogue) by outcome, c = 0.4, eta = 0.2")
    save(fig, f"{FIGURES}/e4_recall_time.png")

    # recall time vs cue corruption, by outcome (pooled loads), c = 0.4
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.0))
    for j, m in enumerate(models):
        ax = axes[j]
        for lab, cond, col in [("true", (0, 0), STATE_COLORS["target"]),
                               ("false (mem cue)", (0, 2), STATE_COLORS["prototype"])]:
            ys = []
            for e_i in range(len(d["etas"])):
                R = d["rows"][j, ci, e_i]
                mask = (R[..., 0] == cond[0]) & (R[..., 1] == cond[1])
                ys.append(R[..., 2][mask].mean() if mask.sum() >= 5 else np.nan)
            ax.plot(d["etas"], ys, "o-", color=col, label=lab)
        ax.set_xlabel(r"cue corruption $\eta$")
        ax.set_title(m)
        if j == 0:
            ax.set_ylabel("mean recall time")
            ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/e4_time_vs_eta.png")


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
