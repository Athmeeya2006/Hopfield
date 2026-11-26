"""E3: blending, beta and temperature (H3).

(a) M3: sweep beta and c and map three regimes -- single memory recall, a
    blend of memories, and prototype recall.
(b) M1 and M2 (n = 3, 4, exp) with Glauber temperature T: the same map in
    the (T, c) plane.  Each run: 50 Glauber sweeps at temperature T followed
    by a zero-temperature quench to the nearest fixed point; the quenched
    state is classified (which basin the noisy dynamics ended in) and the
    raw noisy overlap is also recorded.
(c) Does moderate noise help escape wrong attractors before it destroys
    recall?  Random memories near capacity, retrieval vs T.
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np
from scipy import stats as sstats

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.metrics import OTHER, PROTO, SPURIOUS, TARGET, classify, overlap, participation_ratio
from hopfield.models import Classical, ContinuousExp, Dense
from hopfield.patterns import flip_bits, prototype_patterns, random_patterns, take_per_prototype
from hopfield.seeds import int_seed, rng

N = 100
K = 3
ETA = 0.1
BETAS = np.round(np.logspace(-2.5, 0, 16), 5)
CS_M3 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
PS = [4, 16, 64]
P_MAIN = 16
TS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
CS_BIN = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
BIN_MODELS = ["M1-Hebb", "M2-n3", "M2-n4", "M2-exp"]
GLAUBER_SWEEPS = 50
QUENCH = 50
# (c) noise escape: random memories near capacity
ESC_T = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
ESC = {"M1-Hebb": dict(N=200, P=[20, 24, 28, 32], eta=0.2),
       "M2-n3": dict(N=64, P=[60, 80, 100, 130], eta=0.2)}
if QUICK:
    BETAS = BETAS[::4]
    CS_M3 = CS_M3[::3]
    TS = TS[::3]
    CS_BIN = CS_BIN[::2]


def _structured(s, c, p):
    protos, mems = prototype_patterns(rng("E3", s, int(round(c * 100)), p), K, p, N, c)
    xi, labels = take_per_prototype(mems, p)
    cr = rng("E3-cue", s, int(round(c * 100)), p)
    targets = cr.choice(len(xi), size=N_CUES, replace=len(xi) < N_CUES)
    cues = np.stack([flip_bits(cr, xi[t], ETA) for t in targets])
    return protos, xi, labels, targets, cues


def task_m3(args):
    c, p, s = args
    protos, xi, labels, targets, cues = _structured(s, c, p)
    out = np.zeros((len(BETAS), 5))  # target, other, proto, blend, mean PR
    for ib, b in enumerate(BETAS):
        X, _, W = ContinuousExp(xi, b).recall_batch(cues)
        cats = np.array([classify(X[j], t, xi, protos[labels[t]])[0] for j, t in enumerate(targets)])
        for k in range(4):
            out[ib, k] = np.mean(cats == k)
        out[ib, 4] = participation_ratio(W).mean()
    return out


def bin_model(name, xi):
    if name == "M1-Hebb":
        return Classical.hebbian(xi)
    return Dense(xi, {"M2-n3": 3, "M2-n4": 4, "M2-exp": "exp"}[name])


def task_bin(args):
    name, c, p, s = args
    protos, xi, labels, targets, cues = _structured(s, c, p)
    m = bin_model(name, xi)
    out = np.zeros((len(TS), 6))  # 4 category fractions (quenched), raw m_target, raw m_proto
    for it, T in enumerate(TS):
        cats, mr, mp = [], [], []
        for j, t in enumerate(targets):
            r = m.recall(cues[j], seed=int_seed("E3-dyn", s, int(c * 100), p, it, j), T=T,
                         max_sweeps=GLAUBER_SWEEPS, quench_sweeps=QUENCH if T > 0 else 0,
                         refs=np.stack([xi[t], protos[labels[t]]]), record=True)
            cats.append(classify(r.state, t, xi, protos[labels[t]])[0])
            # raw (noisy) overlaps at the end of the Glauber phase
            mr.append(r.m_traj[r.noise_sweeps, 0])
            mp.append(r.m_traj[r.noise_sweeps, 1])
        cats = np.array(cats)
        for k in range(4):
            out[it, k] = np.mean(cats == k)
        out[it, 4] = np.mean(mr)
        out[it, 5] = np.mean(mp)
    return out


def task_esc(args):
    name, P, s = args
    cfg = ESC[name]
    Nn = cfg["N"]
    xi = random_patterns(rng("E3-esc", name, P, s), P, Nn)
    m = bin_model(name, xi)
    cr = rng("E3-esc-cue", name, P, s)
    targets = cr.choice(P, size=N_CUES, replace=False)
    cues = np.stack([flip_bits(cr, xi[t], cfg["eta"]) for t in targets])
    out = np.zeros((len(ESC_T), 3))  # Pret (quenched), Pret raw (>=0.95 before quench), mean raw m
    for it, T in enumerate(ESC_T):
        ok, okr, mr = [], [], []
        for j, t in enumerate(targets):
            r = m.recall(cues[j], seed=int_seed("E3-esc-dyn", name, P, s, it, j), T=T,
                         max_sweeps=GLAUBER_SWEEPS, quench_sweeps=QUENCH if T > 0 else 0,
                         refs=xi[t:t + 1], record=True)
            ok.append(overlap(r.state, xi[t]) >= 0.95)
            raw = r.m_traj[r.noise_sweeps, 0]
            okr.append(raw >= 0.95)
            mr.append(raw)
        out[it] = [np.mean(ok), np.mean(okr), np.mean(mr)]
    return out


def run():
    warmup()
    with Timer("E3 M3 map"):
        t = [(c, p, s) for c in CS_M3 for p in PS for s in range(N_SETS)]
        o = pmap(task_m3, t, chunksize=4)
    m3 = {}
    for (c, p, s), v in zip(t, o):
        m3.setdefault(str(p), {}).setdefault(str(c), []).append(v)
    with Timer("E3 Glauber maps"):
        t = [(m, c, p, s) for m in BIN_MODELS for c in CS_BIN for p in PS for s in range(N_SETS)]
        o = pmap(task_bin, t)
    binr = {}
    for (m, c, p, s), v in zip(t, o):
        binr.setdefault(m, {}).setdefault(str(p), {}).setdefault(str(c), []).append(v)
    with Timer("E3 noise escape"):
        t = [(m, P, s) for m in ESC for P in ESC[m]["P"] for s in range(N_SETS)]
        o = pmap(task_esc, t)
    esc = {}
    for (m, P, s), v in zip(t, o):
        esc.setdefault(m, {}).setdefault(str(P), []).append(v)
    save_json("e3_raw.json", dict(N=N, K=K, eta=ETA, betas=BETAS, cs_m3=CS_M3, ps=PS, p_main=P_MAIN,
                                  Ts=TS, cs_bin=CS_BIN, bin_models=BIN_MODELS, m3=m3, bin=binr,
                                  esc_T=ESC_T, esc_cfg=ESC, esc=esc))
    analyze()


def _boundary(betas, single):
    """beta at which the single-recall fraction first reaches 0.5 (log interp)."""
    x = np.log(betas)
    for i in range(1, len(single)):
        if single[i - 1] < 0.5 <= single[i]:
            f = (0.5 - single[i - 1]) / (single[i] - single[i - 1])
            return float(np.exp(x[i - 1] + f * (x[i] - x[i - 1])))
    return np.nan if single[-1] < 0.5 else float(betas[0])


def analyze():
    d = load_json("e3_raw.json")
    betas = np.array(d["betas"])
    summ = {"m3_boundaries": {}, "m3_proto_peak": {}, "escape": {}, "glauber": {}}
    for p in d["ps"]:
        for c in d["cs_m3"]:
            arr = np.array(d["m3"][str(p)][str(c)])  # sets x betas x 5
            single = arr[:, :, 0] + arr[:, :, 1]
            summ["m3_boundaries"][f"p{p}_c{c}"] = _boundary(betas, single.mean(0))
            summ["m3_proto_peak"][f"p{p}_c{c}"] = dict(max_proto=float(arr[:, :, 2].mean(0).max()),
                                                      beta_at_max=float(betas[arr[:, :, 2].mean(0).argmax()]))
    # H3 primary: in M3 (p = P_MAIN, c = 0.6) lower beta -> fewer single-memory recalls:
    # per-set Spearman correlation between beta and the single-recall fraction,
    # Wilcoxon signed-rank test of the correlations against 0.
    cprim = "0.6" if "0.6" in d["m3"][str(d["p_main"])] else str(d["cs_m3"][-2])
    arr = np.array(d["m3"][str(d["p_main"])][cprim])
    single = arr[:, :, 0] + arr[:, :, 1]
    rhos = []
    for srow in single:
        if np.ptp(srow) == 0:
            continue
        rhos.append(sstats.spearmanr(betas, srow).statistic)
    rhos = np.array(rhos)
    w = sstats.wilcoxon(rhos) if len(rhos) > 0 and np.any(rhos != 0) else None
    summ["primary_H3"] = dict(
        test="M3, p = %d, c = 0.6: per-set Spearman(beta, fraction single-memory recall); Wilcoxon vs 0" % d["p_main"],
        median_rho=float(np.median(rhos)) if len(rhos) else np.nan, n_sets=int(len(rhos)),
        n_sets_constant=int(single.shape[0] - len(rhos)),
        p=float(w.pvalue) if w is not None else np.nan)
    # Glauber: does higher T increase prototype + spurious (blend) fractions?
    for m in d["bin_models"]:
        for p in d["ps"]:
            for c in d["cs_bin"]:
                arr = np.array(d["bin"][m][str(p)][str(c)])
                mu = arr.mean(0)
                summ["glauber"][f"{m}_p{p}_c{c}"] = dict(
                    target=mu[:, 0], other=mu[:, 1], proto=mu[:, 2], spurious=mu[:, 3],
                    best_T_target=float(np.array(d["Ts"])[mu[:, 0].argmax()]),
                    gain_target=float(mu[:, 0].max() - mu[0, 0]))
    # escape: max over T>0 of Pret minus Pret(T=0), with paired bootstrap CI
    rng_b = np.random.default_rng(3)
    for m in d["esc"]:
        for P, arr in d["esc"][m].items():
            arr = np.array(arr)  # sets x T x 3
            q = arr[:, :, 0]
            gain = q[:, 1:].mean(0).max() - q[:, 0].mean()
            tbest = int(np.argmax(q[:, 1:].mean(0))) + 1
            diffs = q[:, tbest] - q[:, 0]
            bs = [diffs[rng_b.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
            summ["escape"][f"{m}_P{P}"] = dict(Pret_T0=q[:, 0].mean(), Pret_best=q[:, tbest].mean(),
                                               T_best=d["esc_T"][tbest], gain=gain,
                                               gain_lo=np.percentile(bs, 2.5), gain_hi=np.percentile(bs, 97.5),
                                               note="T_best chosen post hoc; CI is optimistic (exploratory)")
    save_json("e3_summary.json", summ)
    print("H3 primary:", summ["primary_H3"])
    print({k: v for k, v in summ["m3_boundaries"].items() if k.startswith(f"p{d['p_main']}")})
    for k, v in summ["escape"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


def plot():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from hopfield.plotting import INK, STATE_COLORS, save, style
    style()
    d = load_json("e3_raw.json")
    betas = np.array(d["betas"])
    cs = np.array(d["cs_m3"])
    # regime maps for M3: dominant category per cell, plus fraction panels
    cmap = ListedColormap([STATE_COLORS["target"], STATE_COLORS["prototype"], "#c9c4b5"])
    fig, axes = plt.subplots(1, len(d["ps"]), figsize=(4.3 * len(d["ps"]), 3.6))
    for ax, p in zip(axes, d["ps"]):
        F = np.array([np.array(d["m3"][str(p)][str(c)]).mean(0) for c in cs])  # c x beta x 5
        single = F[:, :, 0] + F[:, :, 1]
        regimes = np.stack([single, F[:, :, 2], F[:, :, 3]], -1)
        dom = regimes.argmax(-1)
        ax.pcolormesh(np.arange(len(betas) + 1), np.arange(len(cs) + 1), dom, cmap=cmap, vmin=0, vmax=2,
                      shading="flat")
        cs_ = ax.contour(np.arange(len(betas)) + 0.5, np.arange(len(cs)) + 0.5, single, levels=[0.5],
                         colors=INK, linewidths=1)
        ax.set_xticks(np.arange(len(betas))[::3] + 0.5)
        ax.set_xticklabels([f"{b:.3g}" for b in betas[::3]], fontsize=7)
        ax.set_yticks(np.arange(len(cs)) + 0.5)
        ax.set_yticklabels(cs, fontsize=7)
        ax.set_xlabel(r"$\beta$  ($\beta N$ = 100$\beta$)")
        ax.set_ylabel("similarity c")
        ax.set_title(f"M3 regimes, p = {p} per prototype (P = {K * p})")
        ax.grid(False)
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(color=STATE_COLORS["target"], label="single memory"),
                             Patch(color=STATE_COLORS["prototype"], label="prototype"),
                             Patch(color="#c9c4b5", label="blend")], fontsize=7, loc="lower right",
                    frameon=True)
    save(fig, f"{FIGURES}/e3_m3_regime_map.png")

    # fraction curves at c = 0.6 / 0.8 for p_main, with participation ratio
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    avail = [c for c in (0.6, 0.8) if str(c) in d["m3"][str(d["p_main"])]] or [d["cs_m3"][-2], d["cs_m3"][-1]]
    for ax, c in zip(axes[:2], avail):
        arr = np.array(d["m3"][str(d["p_main"])][str(c)])
        mu = arr.mean(0)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        for k, (lab, col) in enumerate([("single memory", STATE_COLORS["target"]),
                                        ("prototype", STATE_COLORS["prototype"]),
                                        ("blend", STATE_COLORS["blend"])]):
            y = mu[:, 0] + mu[:, 1] if k == 0 else mu[:, k + 1]
            ax.plot(betas, y, "o-", color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("fraction of cues")
        ax.set_title(f"M3, c = {c}, p = {d['p_main']}")
        ax.legend(fontsize=7)
    ax = axes[2]
    for c, col in zip([c for c in (0.0, 0.4, 0.8, 0.95) if str(c) in d["m3"][str(d["p_main"])]],
                      plt.cm.Greens(np.linspace(0.4, 0.95, 4))):
        arr = np.array(d["m3"][str(d["p_main"])][str(c)])
        ax.plot(betas, arr[:, :, 4].mean(0), "o-", color=col, label=f"c = {c}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"participation ratio $1/\sum p_\mu^2$")
    ax.set_title("Effective number of memories blended")
    ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/e3_m3_fractions.png")

    # Glauber maps: dominant quenched category in the (T, c) plane
    Ts = np.array(d["Ts"])
    csb = np.array(d["cs_bin"])
    cmap4 = ListedColormap([STATE_COLORS["target"], STATE_COLORS["other memory"],
                            STATE_COLORS["prototype"], "#c9c4b5"])
    fig, axes = plt.subplots(len(d["ps"]), len(d["bin_models"]), figsize=(14, 3 * len(d["ps"])))
    for i, p in enumerate(d["ps"]):
        for j, m in enumerate(d["bin_models"]):
            ax = axes[i, j]
            F = np.array([np.array(d["bin"][m][str(p)][str(c)]).mean(0) for c in csb])  # c x T x 6
            dom = F[:, :, :4].argmax(-1)
            ax.pcolormesh(np.arange(len(Ts) + 1), np.arange(len(csb) + 1), dom, cmap=cmap4, vmin=0, vmax=3)
            for ci in range(len(csb)):
                for ti in range(len(Ts)):
                    ax.text(ti + 0.5, ci + 0.5, f"{F[ci, ti, dom[ci, ti]]:.1f}", ha="center", va="center",
                            fontsize=5, color="white" if dom[ci, ti] in (0, 2) else INK)
            ax.set_xticks(np.arange(len(Ts)) + 0.5)
            ax.set_xticklabels(Ts, fontsize=6, rotation=90)
            ax.set_yticks(np.arange(len(csb)) + 0.5)
            ax.set_yticklabels(csb, fontsize=7)
            ax.set_title(f"{m}, p = {p}", fontsize=9)
            if i == len(d["ps"]) - 1:
                ax.set_xlabel("Glauber temperature T")
            if j == 0:
                ax.set_ylabel("c")
            ax.grid(False)
    axes[0, -1].legend(handles=[Patch(color=STATE_COLORS["target"], label="target"),
                                Patch(color=STATE_COLORS["other memory"], label="other memory"),
                                Patch(color=STATE_COLORS["prototype"], label="prototype"),
                                Patch(color="#c9c4b5", label="spurious / mixture")],
                       fontsize=6, loc="upper right", frameon=True)
    fig.suptitle("Dominant final state (after Glauber phase + zero-T quench); number = its fraction")
    save(fig, f"{FIGURES}/e3_glauber_maps.png")

    # noise escape
    fig, axes = plt.subplots(1, len(d["esc"]), figsize=(5 * len(d["esc"]), 3.4))
    for ax, m in zip(np.atleast_1d(axes), d["esc"]):
        Ps = list(d["esc"][m])
        for P, col in zip(Ps, plt.cm.Greens(np.linspace(0.4, 0.95, len(Ps)))):
            arr = np.array(d["esc"][m][P])
            ax.plot(d["esc_T"], arr[:, :, 0].mean(0), "o-", color=col, label=f"P = {P} (quenched)")
            ax.plot(d["esc_T"], arr[:, :, 1].mean(0), ":", color=col, lw=1)
        ax.set_xlabel("T")
        ax.set_ylabel(r"$P_{ret}$")
        cfg = d["esc_cfg"][m]
        ax.set_title(f"{m}, N = {cfg['N']}, cue eta = {cfg['eta']} (dotted: before quench)")
        ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/e3_noise_escape.png")


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
