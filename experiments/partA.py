"""Part A: validation of the classical model (checks, not findings).

Hebbian and Storkey networks, N in {100, 200, 500, 1000}, load alpha = P/N
from 0.02 to 0.30, cue corruption eta = 0.05.  We check that
  (i)   E never increases under asynchronous updates,
  (ii)  stored patterns are fixed points at low load,
  (iii) the fitted Hebbian critical load approaches 0.138,
and measure cue-noise tolerance (eta_50) and the effect of similarity
between memories (the pilot of the deck).
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np

from common import (FIGURES, N_CUES, N_SETS, Timer, load_json, pmap, save_json,
                    warmup)
from hopfield.learning import hebbian_prefixes, storkey
from hopfield.metrics import ALT_THRESH, THRESH, crossing, overlap
from hopfield.models import Classical
from hopfield.patterns import flip_bits, prototype_patterns, random_patterns, take_per_prototype
from hopfield.seeds import int_seed, rng
from hopfield.stats import bootstrap_ci, bootstrap_stat

NS = [100, 200, 500, 1000]
ALPHAS = np.round(np.arange(0.02, 0.601, 0.02), 2)  # proposal: 0.02-0.30; extended so the Storkey crossing is inside the grid
ETA = 0.05
RULES = ["hebbian", "storkey"]
ETA_GRID = np.round(np.arange(0.0, 0.451, 0.05), 2)
ETA_ALPHAS = [0.05, 0.10]
ETA_N = 500
SIM_N = 200
SIM_CS = [0.0, 0.2, 0.4]
SIM_ALPHAS = np.round(np.arange(0.02, 0.281, 0.02), 2)


def _recall_block(W, xi, targets, cue_rng, eta, dyn_key):
    model = Classical(W)
    N = xi.shape[1]
    out = dict(ret=[], ret90=[], ret99=[], m=[], sweeps=[], time=[], eviol=0)
    for j, t in enumerate(targets):
        cue = flip_bits(cue_rng, xi[t], eta)
        r = model.recall(cue, seed=int_seed(*dyn_key, j), record=True)
        m = overlap(r.state, xi[t])
        out["ret"].append(m >= THRESH)
        out["ret90"].append(m >= ALT_THRESH[0])
        out["ret99"].append(m >= ALT_THRESH[1])
        out["m"].append(m)
        out["sweeps"].append(r.sweeps)
        out["time"].append(r.time)
        out["eviol"] += int(np.sum(np.diff(r.E_traj) > 1e-9))
    return {k: (np.mean(v) if isinstance(v, list) else v) for k, v in out.items()}


def task_capacity(args):
    N, s = args
    Ps = [max(1, int(round(a * N))) for a in ALPHAS]
    xi = random_patterns(rng("A", N, s), max(Ps), N)
    Ws = {"hebbian": hebbian_prefixes(xi, Ps), "storkey": storkey(xi, checkpoints=Ps)}
    res = {r: {k: [] for k in ("ret", "ret90", "ret99", "m", "sweeps", "time", "eviol", "fixed")}
           for r in RULES}
    for ia, P in enumerate(Ps):
        cr = rng("A-cue", N, s, ia)
        targets = cr.choice(P, size=N_CUES, replace=P < N_CUES)
        state = cr.bit_generator.state
        for rule in RULES:
            W = Ws[rule][P]
            cr.bit_generator.state = state  # identical cues for both rules
            o = _recall_block(W, xi[:P], targets, cr, ETA, ("A-dyn", N, s, ia))
            # (ii) fraction of stored patterns that are exact fixed points
            h = xi[:P].astype(float) @ W
            o["fixed"] = np.mean(np.all(h * xi[:P] > 0, axis=1))
            for k in res[rule]:
                res[rule][k].append(o[k])
    return res


def task_eta(args):
    rule, s = args
    N = ETA_N
    out = np.zeros((len(ETA_ALPHAS), len(ETA_GRID)))
    for ia, a in enumerate(ETA_ALPHAS):
        P = int(round(a * N))
        xi = random_patterns(rng("A-eta", N, s, ia), P, N)
        W = storkey(xi) if rule == "storkey" else Classical.hebbian(xi).W
        for ie, eta in enumerate(ETA_GRID):
            cr = rng("A-eta-cue", s, ia, ie)
            targets = cr.choice(P, size=N_CUES, replace=False)
            out[ia, ie] = _recall_block(W, xi, targets, cr, eta, ("A-eta-dyn", s, ia, ie))["ret"]
    return out


def task_similarity(args):
    c, s = args
    N = SIM_N
    Ps = [int(round(a * N)) for a in SIM_ALPHAS]
    _, mems = prototype_patterns(rng("A-sim", s, int(c * 100)), 1, max(Ps), N, c)
    out = []
    for ia, P in enumerate(Ps):
        xi, _ = take_per_prototype(mems, P)
        cr = rng("A-sim-cue", s, ia)
        targets = cr.choice(P, size=N_CUES, replace=P < N_CUES)
        out.append(_recall_block(Classical.hebbian(xi).W, xi, targets, cr, ETA,
                                 ("A-sim-dyn", s, ia))["ret"])
    return out


def run():
    warmup()
    with Timer("Part A capacity"):
        tasks = [(N, s) for N in NS for s in range(N_SETS)]
        outs = pmap(task_capacity, tasks)
    cap = {}
    for (N, s), o in zip(tasks, outs):
        for rule in RULES:
            for k, v in o[rule].items():
                cap.setdefault(rule, {}).setdefault(str(N), {}).setdefault(k, []).append(v)
    with Timer("Part A eta"):
        t2 = [(r, s) for r in RULES for s in range(N_SETS)]
        o2 = pmap(task_eta, t2)
    eta = {r: np.array([o for (rr, _), o in zip(t2, o2) if rr == r]) for r in RULES}
    with Timer("Part A similarity"):
        t3 = [(c, s) for c in SIM_CS for s in range(N_SETS)]
        o3 = pmap(task_similarity, t3)
    sim = {str(c): np.array([o for (cc, _), o in zip(t3, o3) if cc == c]) for c in SIM_CS}
    save_json("partA_raw.json", dict(alphas=ALPHAS, Ns=NS, eta_grid=ETA_GRID,
                                     eta_alphas=ETA_ALPHAS, cap=cap, eta=eta,
                                     sim_alphas=SIM_ALPHAS, sim=sim))
    analyze()


def _alpha50(ret_sets, alphas):
    return crossing(alphas, np.mean(ret_sets, axis=0), 0.5, "down")


def analyze():
    d = load_json("partA_raw.json")
    alphas = np.array(d["alphas"])
    summary = {"checks": {}, "alpha50": {}, "eta50": {}}
    # (i) energy monotone, (ii) fixed points at low load
    tot_viol = 0
    for rule in RULES:
        for N in NS:
            c = d["cap"][rule][str(N)]
            tot_viol += int(np.sum(c["eviol"]))
            fixed = np.array(c["fixed"])
            summary["checks"][f"{rule}_N{N}_fixed_fraction_alpha0.02"] = float(fixed[:, 0].mean())
            summary["checks"][f"{rule}_N{N}_fixed_fraction_alpha0.06"] = float(fixed[:, 2].mean())
    summary["checks"]["energy_increase_events_total"] = tot_viol
    # (iii) alpha_c: 50% crossing per N, then extrapolation in N^-1/2
    a50 = {}
    for rule in RULES:
        rows = []
        boots = []
        for N in NS:
            ret = np.array(d["cap"][rule][str(N)]["ret"])
            est, lo, hi, b = bootstrap_stat([ret], lambda r: _alpha50(r, alphas))
            rows.append(dict(N=N, alpha50=est, lo=lo, hi=hi))
            boots.append(b)
        boots = np.array(boots)  # (len(NS), n_boot)
        x = np.array(NS, float) ** -0.5
        X = np.vstack([np.ones_like(x), x]).T
        coef = np.linalg.lstsq(X, np.array([r["alpha50"] for r in rows]), rcond=None)[0]
        bc = np.array([np.linalg.lstsq(X, boots[:, k], rcond=None)[0] for k in range(boots.shape[1])])
        x1 = np.array(NS, float) ** -1.0
        X1 = np.vstack([np.ones_like(x1), x1]).T
        coef1 = np.linalg.lstsq(X1, np.array([r["alpha50"] for r in rows]), rcond=None)[0]
        a50[rule] = dict(per_N=rows,
                         extrap_inv_sqrtN=dict(alpha_inf=coef[0], slope=coef[1],
                                               lo=np.percentile(bc[:, 0], 2.5),
                                               hi=np.percentile(bc[:, 0], 97.5)),
                         extrap_inv_N_sensitivity=dict(alpha_inf=coef1[0], slope=coef1[1]))
    summary["alpha50"] = a50
    # eta_50
    eg = np.array(d["eta_grid"])
    for rule in RULES:
        arr = np.array(d["eta"][rule])  # sets x alphas x etas
        for ia, a in enumerate(d["eta_alphas"]):
            est, lo, hi, _ = bootstrap_stat([arr[:, ia]], lambda r: crossing(eg, r.mean(0), 0.5, "down"))
            summary["eta50"][f"{rule}_alpha{a}"] = dict(eta50=est, lo=lo, hi=hi)
    # similarity: alpha_50 per c
    summary["similarity_alpha50"] = {}
    sa = np.array(d["sim_alphas"])
    for c, arr in d["sim"].items():
        arr = np.array(arr)
        est, lo, hi, _ = bootstrap_stat([arr], lambda r: crossing(sa, r.mean(0), 0.5, "down"))
        summary["similarity_alpha50"][c] = dict(alpha50=est, lo=lo, hi=hi)
    save_json("partA_summary.json", summary)
    print(summary["checks"])
    for rule in RULES:
        print(rule, a50[rule]["extrap_inv_sqrtN"], [round(r["alpha50"], 4) for r in a50[rule]["per_N"]])
    print(summary["eta50"], summary["similarity_alpha50"])


def num(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        return float(v)
    return v


def plot():
    from hopfield.plotting import INK, band, save, seq, style
    import matplotlib.pyplot as plt
    style()
    d = load_json("partA_raw.json")
    s = load_json("partA_summary.json")
    alphas = np.array(d["alphas"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    cols = seq(len(NS))
    for ax, rule in zip(axes[:2], RULES):
        for N, col in zip(NS, cols):
            ret = np.array(d["cap"][rule][str(N)]["ret"])
            lo, hi = bootstrap_ci(ret)
            ax.plot(alphas, ret.mean(0), "o-", color=col, label=f"N = {N}")
            band(ax, alphas, lo, hi, col)
        ax.axvline(0.138, ls=":", color=INK)
        ax.axvspan(0.30, alphas.max(), color="#d9d6cc", alpha=0.35, lw=0)
        ax.set_title(f"{'Hebbian' if rule == 'hebbian' else 'Storkey'}: capacity (eta = 0.05)")
        ax.set_xlabel(r"load $\alpha = P/N$")
        ax.set_ylabel(r"$P_{ret}$  ($m \geq 0.95$)")
        ax.legend(fontsize=7)
    ax = axes[2]
    for rule, mk in zip(RULES, ("o", "s")):
        rows = [{k: num(v) for k, v in r.items()} for r in s["alpha50"][rule]["per_N"]]
        if not all(np.isfinite(r["alpha50"]) for r in rows):
            continue
        x = np.array([r["N"] for r in rows], float) ** -0.5
        y = [r["alpha50"] for r in rows]
        yerr = [[r["alpha50"] - r["lo"] for r in rows], [r["hi"] - r["alpha50"] for r in rows]]
        ax.errorbar(x, y, yerr=yerr, fmt=mk, color=INK if rule == "hebbian" else "#7a9c6a",
                    label=rule.capitalize(), capsize=2)
        ex = s["alpha50"][rule]["extrap_inv_sqrtN"]
        xx = np.linspace(0, x.max() * 1.05, 20)
        ax.plot(xx, ex["alpha_inf"] + ex["slope"] * xx, "--",
                color=INK if rule == "hebbian" else "#7a9c6a", lw=1)
    ax.axhline(0.138, color="#c27c2c", lw=1, ls=":")
    ax.text(0.002, 0.139, r"$\alpha_c = 0.138$", color="#c27c2c", fontsize=7, va="bottom")
    ex = s["alpha50"]["hebbian"]["extrap_inv_sqrtN"]
    ax.set_title(f"Hebbian $\\alpha_{{50}}$ extrapolated: {ex['alpha_inf']:.3f} "
                 f"[{ex['lo']:.3f}, {ex['hi']:.3f}]")
    ax.set_xlabel(r"$N^{-1/2}$")
    ax.set_ylabel(r"$\alpha_{50}$ (load where $P_{ret} = 0.5$)")
    ax.legend(fontsize=7)
    ax = axes[3]
    sa = np.array(d["sim_alphas"])
    for c, col in zip(d["sim"], seq(3)[::-1]):
        arr = np.array(d["sim"][c])
        lo, hi = bootstrap_ci(arr)
        ax.plot(sa, arr.mean(0), "o-", color=col, label=f"c = {c}")
        band(ax, sa, lo, hi, col)
    ax.set_title(f"Similar memories interfere (Hebbian, N = {SIM_N})")
    ax.set_xlabel(r"load $\alpha$")
    ax.set_ylabel(r"$P_{ret}$")
    ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/partA_capacity.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    eg = np.array(d["eta_grid"])
    for ax, (ia, a) in zip(axes, enumerate(d["eta_alphas"])):
        for rule, col in zip(RULES, (INK, "#7a9c6a")):
            arr = np.array(d["eta"][rule])[:, ia]
            lo, hi = bootstrap_ci(arr)
            ax.plot(eg, arr.mean(0), "o-", color=col, label=rule.capitalize())
            band(ax, eg, lo, hi, col)
        ax.set_title(f"Cue noise tolerance, N = {ETA_N}, alpha = {a}")
        ax.set_xlabel(r"cue corruption $\eta$")
        ax.set_ylabel(r"$P_{ret}$")
        ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/partA_cue_noise.png")


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
