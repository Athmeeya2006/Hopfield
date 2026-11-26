"""E5: lesions and concept cells (H4).

M4 (two-layer, softmax beta = 0.05 and power n = 4): delete a fraction of
memory neurons, delete a fraction of synapses, or add weight noise; compare
with deleting synapses (or adding weight noise) in M1.  Retrieval is measured
for each memory separately.

(a) random memories, N = 200, P = 10, 4 cues per memory: per-memory
    survival curves and an all-or-none index
        AoN = Var_mu(r_mu) / (rbar (1 - rbar))
    (r_mu: retrieval rate of memory mu at a lesion level).  AoN = 1 when
    each memory is either intact or lost; with k cues per memory and
    independent, memory-unspecific failures AoN = 1/k (= 0.25 here).
(b) prototype-structured memories (K = 2, p = 32, c = 0.5): do prototypes
    survive lesions longer than individual memories?

Lesions are nested: one random order of neurons / synapses per pattern set,
and level f removes the first fraction f, so a memory's survival curve is a
property of one progressively damaged network.
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np
from scipy import stats as sstats

from common import FIGURES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.learning import hebbian
from hopfield.metrics import overlap
from hopfield.models import Classical, TwoLayer
from hopfield.patterns import flip_bits, prototype_patterns, random_patterns, take_per_prototype
from hopfield.seeds import int_seed, rng
from hopfield.stats import bootstrap_ci

N = 200
P_A = 10
CUES_PER_MEM = 4
ETA = 0.1
BETA = 0.05
NPOW = 4
DEL = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
NOISE = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
B_K, B_P, B_C = 2, 32, 0.5
B_CUES = 20
CONDS = [("M1-Hebb", "synapse"), ("M1-Hebb", "noise"),
         ("M4-softmax", "neuron"), ("M4-softmax", "synapse"), ("M4-softmax", "noise"),
         ("M4-power", "neuron"), ("M4-power", "synapse"), ("M4-power", "noise")]
if QUICK:
    DEL = DEL[::3]
    NOISE = NOISE[::3]


def levels(lesion):
    return NOISE if lesion == "noise" else DEL


def lesioned_models(model, lesion, xi, s):
    """Yield one damaged model per lesion level (nested damage)."""
    P, Nn = xi.shape
    r = rng("E5-lesion", s, model, lesion, P)
    if model == "M1-Hebb":
        W0 = hebbian(xi)
        iu = np.triu_indices(Nn, 1)
        order = r.permutation(len(iu[0]))
        G = r.standard_normal((Nn, Nn))
        G = np.triu(G, 1)
        G = G + G.T
        sd = W0[iu].std()
        for f in levels(lesion):
            if lesion == "synapse":
                W = W0.copy()
                k = int(round(f * len(order)))
                a, b = iu[0][order[:k]], iu[1][order[:k]]
                W[a, b] = 0.0
                W[b, a] = 0.0
            else:
                W = W0 + f * sd * G
            yield Classical(W)
    else:
        kind = model.split("-")[1]
        W0 = xi.T.astype(np.float64)
        order_n = r.permutation(P)
        order_s = r.permutation(Nn * P)
        G = r.standard_normal((Nn, P))
        for f in levels(lesion):
            W = W0.copy()
            g = np.ones(P)
            if lesion == "neuron":
                g[order_n[:int(round(f * P))]] = 0.0
            elif lesion == "synapse":
                flat = W.ravel()
                flat[order_s[:int(round(f * Nn * P))]] = 0.0
                W = flat.reshape(Nn, P)
            else:
                W = W0 + f * G
            yield TwoLayer(xi, kind, n=NPOW, beta=BETA, g=g, W=W)


def task_a(args):
    model, lesion, s = args
    xi = random_patterns(rng("E5a", s), P_A, N)
    cr = rng("E5a-cue", s)
    cues = [[flip_bits(cr, xi[mu], ETA) for _ in range(CUES_PER_MEM)] for mu in range(P_A)]
    out = np.zeros((len(levels(lesion)), P_A))
    for il, m in enumerate(lesioned_models(model, lesion, xi, s)):
        for mu in range(P_A):
            ok = 0
            for j in range(CUES_PER_MEM):
                r = m.recall(cues[mu][j], seed=int_seed("E5a-dyn", s, mu, j))
                ok += overlap(r.state, xi[mu]) >= 0.95
            out[il, mu] = ok / CUES_PER_MEM
    return out


def task_b(args):
    model, lesion, s = args
    protos, mems = prototype_patterns(rng("E5b", s), B_K, B_P, N, B_C)
    xi, labels = take_per_prototype(mems, B_P)
    cr = rng("E5b-cue", s)
    targets = cr.choice(len(xi), size=B_CUES, replace=False)
    mcues = [flip_bits(cr, xi[t], ETA) for t in targets]
    lk = cr.integers(0, B_K, size=B_CUES)
    lcues = [flip_bits(cr, protos[k], ETA) for k in lk]
    out = np.zeros((len(levels(lesion)), 4))  # item, mem-cue->proto, lure->proto, gist(mem cue -> own category)
    for il, m in enumerate(lesioned_models(model, lesion, xi, s)):
        item = proto_m = proto_l = gist = 0
        for j, t in enumerate(targets):
            r = m.recall(mcues[j], seed=int_seed("E5b-dyn", s, "m", j))
            mo = overlap(r.state, xi)
            item += mo[t] >= 0.95
            pm = overlap(r.state, protos[labels[t]]) >= 0.95
            proto_m += pm
            # gist: recalled the prototype or another stored member of the same category
            sib = (labels == labels[t]) & (np.arange(len(xi)) != t)
            gist += pm or (mo[t] >= 0.95) or bool(np.any(mo[sib] >= 0.95))
        for j, k in enumerate(lk):
            r = m.recall(lcues[j], seed=int_seed("E5b-dyn", s, "l", j))
            proto_l += overlap(r.state, protos[k]) >= 0.95
        out[il] = [item / B_CUES, proto_m / B_CUES, proto_l / B_CUES, gist / B_CUES]
    return out


def run():
    warmup()
    with Timer("E5a"):
        t = [(m, l, s) for m, l in CONDS for s in range(N_SETS)]
        o = pmap(task_a, t)
    a = {}
    for (m, l, s), v in zip(t, o):
        a.setdefault(f"{m}|{l}", []).append(v)
    with Timer("E5b"):
        o = pmap(task_b, t)
    b = {}
    for (m, l, s), v in zip(t, o):
        b.setdefault(f"{m}|{l}", []).append(v)
    save_json("e5_raw.json", dict(N=N, P_a=P_A, cues_per_mem=CUES_PER_MEM, eta=ETA, beta=BETA, n=NPOW,
                                  del_levels=DEL, noise_levels=NOISE, b=dict(K=B_K, p=B_P, c=B_C),
                                  conds=[f"{m}|{l}" for m, l in CONDS], a=a, bres=b))
    analyze()


def aon(r):
    """All-or-none index per lesion level for one set: r is levels x memories."""
    rb = r.mean(1)
    v = r.var(1)
    with np.errstate(all="ignore"):
        x = v / (rb * (1 - rb))
    x[(rb <= 0.1) | (rb >= 0.9)] = np.nan
    return x


def analyze():
    d = load_json("e5_raw.json")
    summ = {"aon": {}, "half_level": {}, "proto_vs_item": {}}
    per_set_aon = {}
    for cond in d["conds"]:
        A = np.array(d["a"][cond])  # sets x levels x P
        idx = np.array([aon(A[s]) for s in range(A.shape[0])])
        with np.errstate(all="ignore"):
            ps = np.nanmean(idx, axis=1)
        per_set_aon[cond] = ps
        ok = np.isfinite(ps)
        lo, hi = bootstrap_ci(ps[ok], np.mean) if ok.sum() > 1 else (np.nan, np.nan)
        summ["aon"][cond] = dict(mean=float(np.nanmean(ps)), lo=float(lo), hi=float(hi), n_sets=int(ok.sum()),
                                 chance_gradual=1 / d["cues_per_mem"])
    # H4 primary: AoN(M4-softmax, neuron deletion) > AoN(M1, synapse deletion), paired Wilcoxon
    x, y = per_set_aon["M4-softmax|neuron"], per_set_aon["M1-Hebb|synapse"]
    ok = np.isfinite(x) & np.isfinite(y)
    w = sstats.wilcoxon(x[ok], y[ok]) if ok.sum() >= 6 else None
    summ["primary_H4"] = dict(test="all-or-none index, M4 (softmax) memory-neuron deletion vs M1 synapse "
                                   "deletion; paired Wilcoxon over sets",
                              mean_M4=float(np.mean(x[ok])), mean_M1=float(np.mean(y[ok])),
                              n_sets=int(ok.sum()), p=float(w.pvalue) if w is not None else np.nan)
    # (b) lesion level at which item recall / prototype recall fall to half of baseline
    for cond in d["conds"]:
        B = np.array(d["bres"][cond]).mean(0)  # levels x 4
        lv = np.array(d["noise_levels"] if cond.endswith("noise") else d["del_levels"])
        res = {}
        for k, lab in enumerate(["item", "memcue_to_proto", "lure_to_proto", "gist"]):
            y = B[:, k]
            base = y[0]
            half = np.nan
            if base > 0.2:
                below = np.where(y < base / 2)[0]
                if below.size:
                    i = below[0]
                    half = float(lv[i - 1] + (base / 2 - y[i - 1]) * (lv[i] - lv[i - 1]) / (y[i] - y[i - 1]))
                else:
                    half = float("inf")
            res[lab] = dict(baseline=float(base), half_level=half)
        summ["proto_vs_item"][cond] = res
    save_json("e5_summary.json", summ)
    print(summ["aon"])
    print("H4 primary:", summ["primary_H4"])
    for k, v in summ["proto_vs_item"].items():
        print(k, v)


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import INK, STATE_COLORS, save, style
    style()
    d = load_json("e5_raw.json")
    conds = d["conds"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, cond in zip(axes.ravel(), conds):
        A = np.array(d["a"][cond])
        lv = np.array(d["noise_levels"] if cond.endswith("noise") else d["del_levels"])
        s0 = A[0]  # one example network: per-memory survival curves
        for mu in range(A.shape[2]):
            ax.plot(lv, s0[:, mu] + (mu - 4.5) * 0.006, "-", color=STATE_COLORS["other memory"], lw=0.9)
        ax.plot(lv, A.mean(axis=(0, 2)), "o-", color=INK, lw=2, label="mean over memories & sets")
        m, l = cond.split("|")
        ax.set_title(f"{m}: {l} {'noise sd' if l == 'noise' else 'deletion'}", fontsize=9)
        ax.set_xlabel("noise sd (x weight sd)" if l == "noise" else "fraction deleted")
        ax.set_ylabel("per-memory retrieval")
        ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("E5a: per-memory survival under lesions (thin lines: the 10 memories of one network)")
    save(fig, f"{FIGURES}/e5_survival.png")

    s = load_json("e5_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    names = list(s["aon"])
    f = lambda v: np.nan if v is None else float(v)  # noqa: E731
    vals = [f(s["aon"][n]["mean"]) for n in names]
    err = [[max(0.0, f(s["aon"][n]["mean"]) - f(s["aon"][n]["lo"])) if np.isfinite(f(s["aon"][n]["lo"])) else 0
            for n in names],
           [max(0.0, f(s["aon"][n]["hi"]) - f(s["aon"][n]["mean"])) if np.isfinite(f(s["aon"][n]["hi"])) else 0
            for n in names]]
    ax.bar(range(len(names)), vals, yerr=err, color=[("#2a7f86" if "softmax" in n else "#b5527a" if "power" in n
                                                      else "#1b3a2a") for n in names], capsize=3)
    ax.axhline(1 / d["cues_per_mem"], ls=":", color=INK)
    ax.text(len(names) - 0.5, 1 / d["cues_per_mem"] + 0.02, "independent failures", fontsize=7, ha="right")
    ax.axhline(1.0, ls="--", color=INK, lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("|", "\n") for n in names], fontsize=7)
    ax.set_ylabel("all-or-none index")
    ax.set_title("Distributed vs concept-cell signatures (mean, 95% CI)")
    ax = axes[1]
    for cond, col in [("M1-Hebb|synapse", "#1b3a2a"), ("M4-softmax|neuron", "#2a7f86"),
                      ("M4-softmax|synapse", "#6fb3b8"), ("M4-power|neuron", "#b5527a")]:
        B = np.array(d["bres"][cond]).mean(0)
        lv = np.array(d["del_levels"])
        ax.plot(lv, B[:, 0], "o-", color=col, label=f"{cond}: item")
        ax.plot(lv, B[:, 2], "s--", color=col, label=f"{cond}: prototype (lure cue)")
    ax.set_xlabel("fraction deleted")
    ax.set_ylabel("recall")
    ax.set_title(f"E5b: items vs prototype (K = {d['b']['K']}, p = {d['b']['p']}, c = {d['b']['c']})")
    ax.legend(fontsize=6, ncol=2)
    save(fig, f"{FIGURES}/e5_aon_prototype.png")


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
