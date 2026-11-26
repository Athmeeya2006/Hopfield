"""E6: attention as associative memory (H5).

M3 is treated as one attention head with key dimension d = 64 and
beta = b / sqrt(d), b in {0.5, 1, 2, 4} (b = 1 is standard attention
scaling).  P key/value pairs form the context; a noisy query (the target
key with a fraction eta of its bits flipped) retrieves
    o = V^T softmax(beta K q).
Varied: context length P, similarity rho between the target key and the
distractor keys (distractor = target key with each bit flipped with
probability (1-rho)/2, so expected overlap rho), and beta.  Retrieval is
correct if the output is closer (cosine) to the target value than to any
other value.  No language model is trained or run.
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np
from scipy import stats as sstats

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_json, pmap, save_json
from hopfield.seeds import rng

D = 64
ETA = 0.1
PS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
BS = [0.5, 1.0, 2.0, 4.0]
ETAS = [0.05, 0.1, 0.2, 0.3]   # query-noise sweep at b = 1
if QUICK:
    PS = PS[::3]


def head(K, V, q, beta):
    z = beta * (K @ q)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return V.T @ p, p


def trial(r, P, rho, beta, eta):
    k0 = 2 * r.integers(0, 2, D) - 1
    flip = r.random((P - 1, D)) < (1 - rho) / 2
    K = np.vstack([k0, np.where(flip, -k0, k0)]).astype(float)
    V = (2 * r.integers(0, 2, (P, D)) - 1).astype(float)
    q = k0.astype(float).copy()
    idx = r.choice(D, int(round(eta * D)), replace=False)
    q[idx] *= -1
    o, p = head(K, V, q, beta)
    cos = V @ o / (np.linalg.norm(o) * np.sqrt(D) + 1e-12)
    return float(np.argmax(cos) == 0), float(cos[0]), float(p[0]), float(1 / np.sum(p ** 2))


def task(args):
    kind, s = args
    if kind == "main":
        out = np.zeros((len(BS), len(RHOS), len(PS), 4))
        for ib, b in enumerate(BS):
            for ir, rho in enumerate(RHOS):
                for ip, P in enumerate(PS):
                    r = rng("E6", s, ib, ir, ip)
                    out[ib, ir, ip] = np.mean([trial(r, P, rho, b / np.sqrt(D), ETA) for _ in range(N_CUES)], 0)
    else:
        out = np.zeros((len(ETAS), len(RHOS), len(PS), 4))
        for ie, eta in enumerate(ETAS):
            for ir, rho in enumerate(RHOS):
                for ip, P in enumerate(PS):
                    r = rng("E6-eta", s, ie, ir, ip)
                    out[ie, ir, ip] = np.mean([trial(r, P, rho, 1 / np.sqrt(D), eta) for _ in range(N_CUES)], 0)
    return out


def run():
    with Timer("E6"):
        t = [(k, s) for k in ("main", "eta") for s in range(N_SETS)]
        o = pmap(task, t)
    main = np.array([v for (k, _), v in zip(t, o) if k == "main"])
    eta = np.array([v for (k, _), v in zip(t, o) if k == "eta"])
    save_json("e6_raw.json", dict(d=D, eta=ETA, Ps=PS, rhos=RHOS, bs=BS, etas=ETAS, main=main, eta_sweep=eta))
    analyze()


def _p50(Ps, acc):
    x = np.log(Ps)
    for i in range(1, len(acc)):
        if acc[i - 1] >= 0.5 > acc[i]:
            return float(np.exp(x[i - 1] + (acc[i - 1] - 0.5) / (acc[i - 1] - acc[i]) * (x[i] - x[i - 1])))
    return float("inf") if acc[-1] >= 0.5 else float("-inf")


def analyze():
    d = load_json("e6_raw.json")
    M = np.array(d["main"])  # sets x b x rho x P x 4
    Ps = np.array(d["Ps"], float)
    summ = {"P50": {}}
    for ib, b in enumerate(d["bs"]):
        for ir, rho in enumerate(d["rhos"]):
            acc = M[:, ib, ir, :, 0].mean(0)
            summ["P50"][f"b{b}_rho{rho}"] = _p50(Ps, acc)
    ib = d["bs"].index(1.0)
    i0, i8 = d["rhos"].index(0.0), d["rhos"].index(0.8)
    a0 = M[:, ib, i0, :, 0].mean(1)
    a8 = M[:, ib, i8, :, 0].mean(1)
    w = sstats.wilcoxon(a0, a8)
    summ["primary_H5"] = dict(test="beta = 1/sqrt(d): per-set mean accuracy over context lengths, key "
                                   "similarity rho = 0 vs rho = 0.8 (paired Wilcoxon)",
                              mean_rho0=float(a0.mean()), mean_rho08=float(a8.mean()), p=float(w.pvalue))
    # accuracy decreases with context length: per-set Spearman at rho = 0.6
    i6 = d["rhos"].index(0.6)
    rh = [sstats.spearmanr(Ps, M[s, ib, i6, :, 0]).statistic for s in range(M.shape[0])
          if np.ptp(M[s, ib, i6, :, 0]) > 0]
    summ["context_length_trend_rho0.6"] = dict(median_spearman=float(np.median(rh)), n=len(rh))
    save_json("e6_summary.json", summ)
    print(summ)


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import INK, save, seq, style
    style()
    d = load_json("e6_raw.json")
    M = np.array(d["main"])
    Ps = np.array(d["Ps"])
    cols = seq(len(d["rhos"]))
    fig, axes = plt.subplots(1, len(d["bs"]), figsize=(4 * len(d["bs"]), 3.3), sharey=True)
    for ax, (ib, b) in zip(axes, enumerate(d["bs"])):
        for ir, (rho, col) in enumerate(zip(d["rhos"], cols)):
            a = M[:, ib, ir, :, 0]
            ax.plot(Ps, a.mean(0), "o-", color=col, label=f"rho = {rho}", ms=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length P")
        ax.set_title(fr"$\beta = {b}/\sqrt{{d}}$")
        ax.axhline(0.5, ls=":", color=INK, lw=0.8)
    axes[0].set_ylabel("retrieval accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle("E6: one attention head (d = 64) as associative memory, query noise eta = 0.1")
    save(fig, f"{FIGURES}/e6_accuracy.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    ib = d["bs"].index(1.0)
    for ir, (rho, col) in enumerate(zip(d["rhos"], cols)):
        axes[0].plot(Ps, M[:, ib, ir, :, 2].mean(0), "o-", color=col, ms=3, label=f"rho = {rho}")
        axes[1].plot(Ps, M[:, ib, ir, :, 3].mean(0), "o-", color=col, ms=3)
    axes[0].set_ylabel("attention weight on target")
    axes[1].set_ylabel("participation ratio (keys blended)")
    for ax in axes[:2]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length P")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[0].set_title(r"Blending at $\beta = 1/\sqrt{d}$")
    E = np.array(d["eta_sweep"])
    ir = d["rhos"].index(0.6)
    for ie, (eta, col) in enumerate(zip(d["etas"], seq(len(d["etas"])))):
        axes[2].plot(Ps, E[:, ie, ir, :, 0].mean(0), "o-", color=col, ms=3, label=f"query eta = {eta}")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("context length P")
    axes[2].set_ylabel("retrieval accuracy")
    axes[2].set_title(r"Query noise, rho = 0.6, $\beta = 1/\sqrt{d}$")
    axes[2].legend(fontsize=7)
    save(fig, f"{FIGURES}/e6_blending.png")


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
