"""E1: capacity across models (H1).

M1 (Hebbian, Storkey), M2 (n = 3, 4, exp) and M3 (beta = 1 and beta = 1/sqrt(N))
are run on identical random patterns and cues.  For each N we find the
largest P with retrieval probability >= 0.5 (P_max), then fit log P_max
against log N and compare slopes with the theoretical N^(n-1) and
exponential laws.

Cue corruption eta = 0.125 (exactly N/8 bits flipped; all N are multiples
of 8 so the corruption is identical across N).  P_max is located by an
adaptive geometric ladder (x2 to bracket the 0.5 crossing, then x2^(1/4)
refinement) and interpolated in log P.  Every ladder point uses all
pattern sets x cues.
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.learning import storkey
from hopfield.metrics import ALT_THRESH, THRESH, log_crossing, overlap
from hopfield.models import Classical, ContinuousExp, Dense
from hopfield.patterns import flip_bits, random_patterns
from hopfield.seeds import int_seed, rng
from hopfield.stats import N_BOOT, boot_p_value

ETA = 0.125
NGRID = {
    "M1-Hebb": [16, 32, 64, 128, 256, 512, 1000],
    "M1-Storkey": [16, 32, 64, 128, 256, 512, 1000],
    "M2-n3": [16, 24, 32, 48, 64, 96],
    "M2-n4": [16, 24, 32, 40, 48, 56],
    "M2-exp": [8, 16, 24, 32],
    "M3-b1": [8, 16, 24, 32],
    "M3-bsqrt": [8, 16, 24, 32, 64, 128],
}
if QUICK:
    NGRID = {k: v[:3] for k, v in NGRID.items()}
# E1_PART splits the run across machines: "m1", "m2poly", "exp" (default: all)
PARTS = {"m1": ["M1-Hebb", "M1-Storkey"], "m2poly": ["M2-n3", "M2-n4"], "exp": ["M2-exp", "M3-b1", "M3-bsqrt"]}
import os as _os  # noqa: E402
PART = _os.environ.get("E1_PART", "all")
THEORY_SLOPE = {"M1-Hebb": 1, "M1-Storkey": 1, "M2-n3": 2, "M2-n4": 3}
P_CAP = 400_000
CHUNK = 4096


def guess(model, N):
    return {
        "M1-Hebb": 0.13 * N, "M1-Storkey": 0.25 * N, "M2-n3": 0.05 * N ** 2,
        "M2-n4": 0.01 * N ** 3, "M2-exp": np.exp(0.33 * N), "M3-b1": np.exp(0.33 * N),
        "M3-bsqrt": 0.5 * N,
    }[model]


def bank(N, s, P):
    """First P random patterns of pattern set s at size N (prefix-consistent)."""
    nch = (P + CHUNK - 1) // CHUNK
    rows = [random_patterns(rng("E1", N, s, ch), CHUNK, N) for ch in range(nch)]
    return np.ascontiguousarray(np.concatenate(rows)[:P])


def task(args):
    model, N, P, s = args
    xi = bank(N, s, P)
    cr = rng("E1-cue", N, s, P)
    targets = cr.choice(P, size=N_CUES, replace=P < N_CUES)
    cues = np.stack([flip_bits(cr, xi[t], ETA) for t in targets])
    if model == "M1-Hebb":
        m = Classical.hebbian(xi)
    elif model == "M1-Storkey":
        m = Classical(storkey(xi))
    elif model.startswith("M2"):
        m = Dense(xi, {"M2-n3": 3, "M2-n4": 4, "M2-exp": "exp"}[model])
    else:
        beta = 1.0 if model == "M3-b1" else 1.0 / np.sqrt(N)
        m = ContinuousExp(xi, beta)
    if model.startswith("M3"):
        X, iters, _ = m.recall_batch(cues)
        ms = np.array([overlap(X[j], xi[t]) for j, t in enumerate(targets)])
        sweeps = iters
    else:
        ms, sweeps = [], []
        for j, t in enumerate(targets):
            r = m.recall(cues[j], seed=int_seed("E1-dyn", N, s, P, j))
            ms.append(overlap(r.state, xi[t]))
            sweeps.append(r.time)
        ms = np.array(ms)
    return dict(ret=np.mean(ms >= THRESH), ret90=np.mean(ms >= ALT_THRESH[0]),
                ret99=np.mean(ms >= ALT_THRESH[1]), m=ms.mean(), time=np.mean(sweeps))


def evaluate(model, N, P):
    outs = pmap(task, [(model, N, P, s) for s in range(N_SETS)])
    return {k: [o[k] for o in outs] for k in outs[0]}


def ladder(model, N, log):
    """Adaptive search for the 0.5 crossing of mean P_ret(P)."""
    res = {}
    cap = min(P_CAP, 2 ** min(N, 40))

    def ev(P):
        P = int(max(1, min(cap, round(P))))
        if P not in res:
            res[P] = evaluate(model, N, P)
            log(f"  {model} N={N} P={P}: Pret={np.mean(res[P]['ret']):.3f}")
        return P, np.mean(res[P]["ret"])

    P, r = ev(max(2, guess(model, N)))
    if r >= 0.5:
        lo = P
        while True:
            if P >= cap:
                return res  # censored above
            P, r = ev(P * 2)
            if r < 0.5:
                hi = P
                break
            lo = P
    else:
        hi = P
        while True:
            if P <= 1:
                return res
            P, r = ev(P / 2)
            if r >= 0.5:
                lo = P
                break
            hi = P
    for k in (1, 2, 3):
        ev(lo * (hi / lo) ** (k / 4))
    ev(hi * 2 ** 0.5)  # one point beyond, for the curves
    if lo / 2 ** 0.5 >= 1:
        ev(lo / 2 ** 0.5)
    return res


def run():
    warmup()
    out = {}
    models = PARTS[PART] if PART in PARTS else list(NGRID)
    with Timer("E1"):
        for model, Ns in ((m, NGRID[m]) for m in models):
            for N in Ns:
                res = ladder(model, N, lambda m: print(m, flush=True))
                out.setdefault(model, {})[str(N)] = {str(P): v for P, v in sorted(res.items())}
                save_json(f"e1_raw_{PART}.json", dict(eta=ETA, grid=NGRID, data=out))
    analyze()


def _pmax(curve_P, ret_sets, idx=None):
    r = ret_sets if idx is None else ret_sets[:, idx]
    return log_crossing(curve_P, r.mean(axis=1), 0.5, "down")


def merge_raw():
    """Merge e1_raw_<part>.json files into e1_raw.json."""
    import glob
    data = {}
    for f in sorted(glob.glob(f"{common.RESULTS}/e1_raw_*.json")):
        data.update(load_json(f.split("/")[-1])["data"])
    order = [m for m in NGRID if m in data]
    save_json("e1_raw.json", dict(eta=ETA, grid=NGRID, data={m: data[m] for m in order}))


def analyze():
    merge_raw()
    d = load_json("e1_raw.json")["data"]
    brng = np.random.default_rng(7)
    allN = sorted({int(N) for m in d for N in d[m]})
    boot_idx = {N: brng.integers(0, N_SETS, size=(N_BOOT, N_SETS)) for N in allN}
    summ = {}
    for model, byN in d.items():
        rows = []
        boots = []
        for N, byP in byN.items():
            Ps = np.array(sorted(int(p) for p in byP))
            ret = np.array([byP[str(p)]["ret"] for p in Ps])  # P x sets
            est = _pmax(Ps, ret)
            b = np.array([_pmax(Ps, ret, ix) for ix in boot_idx[int(N)]])
            rows.append(dict(N=int(N), Pmax=est, lo=np.nanpercentile(b, 2.5),
                             hi=np.nanpercentile(b, 97.5),
                             Pmax90=_pmax(Ps, np.array([byP[str(p)]["ret90"] for p in Ps])),
                             Pmax99=_pmax(Ps, np.array([byP[str(p)]["ret99"] for p in Ps]))))
            boots.append(b)
        boots = np.array(boots)  # N x boot
        Ns = np.array([r["N"] for r in rows], float)
        Pm = np.array([r["Pmax"] for r in rows], float)
        ok = np.isfinite(Pm) & (Pm > 0)
        fit = {}
        if ok.sum() >= 2:
            # power law: log Pmax = a + k log N
            k, a = np.polyfit(np.log(Ns[ok]), np.log(Pm[ok]), 1)
            kb = np.array([np.polyfit(np.log(Ns[ok]), np.log(boots[ok, j]), 1)[0]
                           for j in range(boots.shape[1]) if np.all(np.isfinite(boots[ok, j]))])
            # exponential law: log Pmax = a + alpha N
            al, a2 = np.polyfit(Ns[ok], np.log(Pm[ok]), 1)
            alb = np.array([np.polyfit(Ns[ok], np.log(boots[ok, j]), 1)[0]
                            for j in range(boots.shape[1]) if np.all(np.isfinite(boots[ok, j]))])
            fit = dict(loglog_slope=k, loglog_lo=np.percentile(kb, 2.5), loglog_hi=np.percentile(kb, 97.5),
                       loglog_intercept=a, exp_rate=al, exp_lo=np.percentile(alb, 2.5),
                       exp_hi=np.percentile(alb, 97.5), exp_intercept=a2,
                       theory_slope=THEORY_SLOPE.get(model))
        summ[model] = dict(rows=rows, fit=fit, boots=boots)
    # H1 primary test: log-log slope of M2-n4 exceeds that of M1-Hebb (paired bootstrap)
    primary = None
    if "M2-n4" in summ and "M1-Hebb" in summ:
        def slopes(model, j):
            rows, bt = summ[model]["rows"], summ[model]["boots"]
            Ns = np.log([r["N"] for r in rows])
            y = bt[:, j]
            if not np.all(np.isfinite(y)) or np.any(y <= 0):
                return np.nan
            return np.polyfit(Ns, np.log(y), 1)[0]
        diff = np.array([slopes("M2-n4", j) - slopes("M1-Hebb", j) for j in range(N_BOOT)])
        est = summ["M2-n4"]["fit"]["loglog_slope"] - summ["M1-Hebb"]["fit"]["loglog_slope"]
        primary = dict(test="log-log capacity slope, M2 (n=4) minus M1 Hebbian (paired bootstrap over sets)",
                       estimate=est, lo=np.nanpercentile(diff, 2.5), hi=np.nanpercentile(diff, 97.5),
                       p=boot_p_value(diff, 0.0))
    for m in summ:
        summ[m].pop("boots")
    # Demircigil et al. bound for the exponential model with cue noise rho = eta
    x = 1 - 2 * ETA
    I = 0.5 * ((1 + x) * np.log(1 + x) + (1 - x) * np.log(1 - x))
    save_json("e1_summary.json", dict(models=summ, primary_H1=primary,
                                      demircigil_rate_bound=I / 2, eta=ETA))
    for m, v in summ.items():
        print(m, [(r["N"], round(r["Pmax"], 1) if np.isfinite(r["Pmax"]) else r["Pmax"]) for r in v["rows"]],
              {k: (round(v2, 3) if isinstance(v2, float) else v2) for k, v2 in v["fit"].items()})
    print("H1 primary:", primary)


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import CAT, INK, save, style
    style()
    d = load_json("e1_raw.json")["data"]
    s = load_json("e1_summary.json")
    num = lambda v: float(v) if v is not None else np.nan  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    for model, v in s["models"].items():
        rows = v["rows"]
        N = np.array([r["N"] for r in rows], float)
        P = np.array([num(r["Pmax"]) for r in rows])
        lo = np.array([num(r["lo"]) for r in rows])
        hi = np.array([num(r["hi"]) for r in rows])
        col = CAT.get(model, CAT.get(model.split("-b")[0], INK))
        if model == "M3-bsqrt":
            col = "#a38bd0"
        ax.errorbar(N, P, yerr=[P - lo, hi - P], fmt="o-", color=col, capsize=2, label=model, ms=4)
    NN = np.logspace(np.log10(8), 3, 50)
    ax.plot(NN, 0.138 * NN, ":", color=INK, lw=1)
    ax.text(300, 0.138 * 300 * 0.55, r"$0.138N$", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$P_{max}$ ($P_{ret} \geq 0.5$)")
    ax.set_title(r"E1: capacity vs N (log-log), $\eta = 0.125$")
    ax.legend(fontsize=7, ncol=2)
    # slopes vs theory
    ax = axes[1]
    names, ks, los, his, th = [], [], [], [], []
    for model, v in s["models"].items():
        f = v["fit"]
        if not f:
            continue
        names.append(model)
        ks.append(f["loglog_slope"])
        los.append(f["loglog_slope"] - f["loglog_lo"])
        his.append(f["loglog_hi"] - f["loglog_slope"])
        th.append(f["theory_slope"] if f["theory_slope"] is not None else np.nan)
    x = np.arange(len(names))
    ax.bar(x, ks, yerr=[los, his], color=[CAT.get(n, "#a38bd0") for n in names], capsize=3, alpha=0.85)
    ax.scatter(x, th, marker="_", s=400, color="k", zorder=5, label="theory $n-1$")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"fitted slope $d\log P_{max}/d\log N$")
    ax.set_title("Scaling exponents (95% bootstrap CI)")
    ax.legend(fontsize=7)
    # exponential models: log P vs N
    ax = axes[2]
    for model in ("M2-exp", "M3-b1", "M2-n4", "M2-n3"):
        if model not in s["models"]:
            continue
        rows = s["models"][model]["rows"]
        N = np.array([r["N"] for r in rows], float)
        P = np.array([num(r["Pmax"]) for r in rows])
        ax.plot(N, P, "o-", color=CAT.get(model, "#6c4f9c"), label=model)
        f = s["models"][model]["fit"]
        if model in ("M2-exp", "M3-b1") and f:
            ax.plot(N, np.exp(f["exp_intercept"] + f["exp_rate"] * N), "--", lw=1,
                    color=CAT.get(model, "#6c4f9c"))
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$P_{max}$ (log scale)")
    fe = s["models"].get("M2-exp", {}).get("fit", {})
    if fe:
        ax.set_title(f"Exponential law: rate {fe['exp_rate']:.3f} "
                     f"[{fe['exp_lo']:.3f}, {fe['exp_hi']:.3f}] per neuron")
    ax.legend(fontsize=7)
    save(fig, f"{FIGURES}/e1_capacity.png")

    # retrieval curves per model
    models = list(d)
    fig, axes = plt.subplots(2, (len(models) + 1) // 2, figsize=(15, 6.5))
    for ax, model in zip(axes.ravel(), models):
        Ns = sorted(d[model], key=int)
        cols = plt.cm.Greens(np.linspace(0.35, 0.95, len(Ns)))
        for N, col in zip(Ns, cols):
            Ps = np.array(sorted(int(p) for p in d[model][N]))
            r = np.array([np.mean(d[model][N][str(p)]["ret"]) for p in Ps])
            ax.plot(Ps, r, "o-", color=col, label=f"N={N}", ms=3)
        ax.axhline(0.5, color=INK, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(model)
        ax.set_xlabel("P")
        ax.set_ylabel(r"$P_{ret}$")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    save(fig, f"{FIGURES}/e1_retrieval_curves.png")


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
