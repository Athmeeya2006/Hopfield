"""E7: forgetting and recency (H6).

A long stream of memories (S = 3N) is stored one by one with clipping
(Parisi) or decay (Mezard-Nadal-Toulouse) in M1, and with decaying
memory-neuron strengths (g_mu <- lambda g_mu at every new memory) in M4.
After the stream every memory is cued and retrieval is plotted against its
age (number of memories stored after it).  Memory span = age at which
retrieval falls below 0.5 (also reported at 0.9).  The unbounded network is
the comparison.

(b) Prototype-structured stream ("events"): blocks of B = 20 memories, each
block built around its own unstored prototype (c = 0.5).  Does the
prototype (gist) survive after its individual memories fade?
"""
from __future__ import annotations

import argparse

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np
from scipy import stats as sstats

from common import FIGURES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.learning import stream_clipped, stream_decay
from hopfield.metrics import overlap
from hopfield.models import Classical, TwoLayer
from hopfield.patterns import flip_bits, prototype_patterns, random_patterns
from hopfield.seeds import int_seed, rng

N = 200
S = 600
ETA = 0.1
BETA = 0.05
NPOW = 4
CONDS = ([("M1", "clip", A) for A in (0.3, 0.4, 0.6, 1.0, 2.0)] +
         [("M1", "decay", l) for l in (0.90, 0.95, 0.98, 0.99, 0.995)] +
         [("M1", "none", 0)] +
         [("M4-softmax", "decay", l) for l in (0.5, 0.8, 0.9, 0.95, 0.98, 1.0)] +
         [("M4-power", "decay", l) for l in (0.5, 0.8, 0.9, 0.95, 0.98, 1.0)])
B_BLOCK, B_C = 20, 0.5
B_CONDS = ([("M1", "decay", l) for l in (0.95, 0.98, 0.99)] + [("M1", "clip", A) for A in (0.4, 1.0)] +
           [("M4-softmax", "decay", l) for l in (0.9, 0.95, 0.98)] +
           [("M4-power", "decay", l) for l in (0.9, 0.95, 0.98)])
LURES_PER_BLOCK = 2
if QUICK:
    S = 200
    CONDS = CONDS[::3]
    B_CONDS = B_CONDS[::3]


def build(model, rule, par, xi):
    if model == "M1":
        if rule == "clip":
            return Classical(stream_clipped(xi, par))
        if rule == "decay":
            return Classical(stream_decay(xi, par))
        return Classical(stream_clipped(xi, np.inf))
    P = len(xi)
    g = par ** np.arange(P - 1, -1, -1, dtype=float)  # memory stored at position mu has age P-1-mu
    return TwoLayer(xi, model.split("-")[1], n=NPOW, beta=BETA, g=g)


def task_a(args):
    ci, s = args
    model, rule, par = CONDS[ci]
    xi = random_patterns(rng("E7", s), S, N)
    m = build(model, rule, par, xi)
    cr = rng("E7-cue", s)
    out = np.zeros(S)  # indexed by age
    for mu in range(S):
        cue = flip_bits(cr, xi[mu], ETA)
        r = m.recall(cue, seed=int_seed("E7-dyn", s, mu))
        out[S - 1 - mu] = overlap(r.state, xi[mu]) >= 0.95
    return out


def task_b(args):
    ci, s = args
    model, rule, par = B_CONDS[ci]
    nb = S // B_BLOCK
    protos, mems = prototype_patterns(rng("E7b", s), nb, B_BLOCK, N, B_C)
    xi = mems.reshape(nb * B_BLOCK, N)  # block after block
    m = build(model, rule, par, xi)
    cr = rng("E7b-cue", s)
    item = np.zeros(S)
    item_to_proto = np.zeros(S)
    proto = np.zeros(nb)
    for mu in range(S):
        r = m.recall(flip_bits(cr, xi[mu], ETA), seed=int_seed("E7b-dyn", s, mu))
        item[S - 1 - mu] = overlap(r.state, xi[mu]) >= 0.95
        item_to_proto[S - 1 - mu] = overlap(r.state, protos[mu // B_BLOCK]) >= 0.95
    for b in range(nb):
        ok = 0
        for j in range(LURES_PER_BLOCK):
            r = m.recall(flip_bits(cr, protos[b], ETA), seed=int_seed("E7b-lure", s, b, j))
            ok += overlap(r.state, protos[b]) >= 0.95
        proto[nb - 1 - b] = ok / LURES_PER_BLOCK  # indexed by block age
    return np.concatenate([item, item_to_proto, proto])


def run():
    warmup()
    with Timer("E7a"):
        t = [(ci, s) for ci in range(len(CONDS)) for s in range(N_SETS)]
        o = pmap(task_a, t)
    a = {}
    for (ci, s), v in zip(t, o):
        a.setdefault("|".join(map(str, CONDS[ci])), []).append(v)
    with Timer("E7b"):
        t = [(ci, s) for ci in range(len(B_CONDS)) for s in range(N_SETS)]
        o = pmap(task_b, t)
    b = {}
    for (ci, s), v in zip(t, o):
        b.setdefault("|".join(map(str, B_CONDS[ci])), []).append(v)
    save_json("e7_raw.json", dict(N=N, S=S, eta=ETA, beta=BETA, n=NPOW, block=B_BLOCK, c=B_C,
                                  conds=["|".join(map(str, c)) for c in CONDS],
                                  b_conds=["|".join(map(str, c)) for c in B_CONDS], a=a, b=b))
    analyze()


def span(curve, level=0.5):
    """Number of most recent memories retrieved with probability >= level:
    first age at which the (smoothed) curve drops below level."""
    k = 5
    sm = np.convolve(curve, np.ones(k) / k, mode="same")
    sm[:k // 2] = curve[:k // 2]
    below = np.where(sm < level)[0]
    return int(below[0]) if below.size else len(curve)


def analyze():
    d = load_json("e7_raw.json")
    summ = {"span": {}, "gist": {}}
    brng = np.random.default_rng(5)
    for cond in d["conds"]:
        A = np.array(d["a"][cond])  # sets x age
        sp = span(A.mean(0))
        bs = [span(A[brng.integers(0, len(A), len(A))].mean(0)) for _ in range(1000)]
        summ["span"][cond] = dict(span50=sp, lo=int(np.percentile(bs, 2.5)), hi=int(np.percentile(bs, 97.5)),
                                  span90=span(A.mean(0), 0.9), recent20=float(A[:, :20].mean()),
                                  overall=float(A.mean()))
    # H6 primary: bounded (decay lambda = 0.98) vs unbounded M1: retrieval of the 20 most recent memories
    x = np.array(d["a"]["M1|decay|0.98"])[:, :20].mean(1) if "M1|decay|0.98" in d["a"] else None
    y = np.array(d["a"]["M1|none|0"])[:, :20].mean(1) if "M1|none|0" in d["a"] else None
    if x is not None and y is not None:
        w = sstats.wilcoxon(x, y) if np.any(x != y) else None
        summ["primary_H6"] = dict(test="M1 retrieval of the 20 most recent memories after a 3N stream: decay "
                                       "lambda = 0.98 vs unbounded Hebbian (paired Wilcoxon over sets)",
                                  mean_bounded=float(x.mean()), mean_unbounded=float(y.mean()),
                                  p=float(w.pvalue) if w is not None else np.nan)
    nb = d["S"] // d["block"]
    for cond in d["b_conds"]:
        B = np.array(d["b"][cond])
        item = B[:, :d["S"]]
        i2p = B[:, d["S"]:2 * d["S"]]
        proto = B[:, 2 * d["S"]:]
        # per block age: mean item recall of that block's memories
        item_block = item.reshape(len(B), nb, d["block"]).mean(2)  # index = block age
        sp_item = span(item_block.mean(0))
        sp_proto = span(proto.mean(0))
        diff = proto.mean(1) - item_block.mean(1)
        w = sstats.wilcoxon(proto.mean(1), item_block.mean(1)) if np.any(diff != 0) else None
        summ["gist"][cond] = dict(item_span_blocks=sp_item, proto_span_blocks=sp_proto,
                                  mean_item=float(item_block.mean()), mean_proto=float(proto.mean()),
                                  mean_item_to_proto=float(i2p.mean()),
                                  p_proto_vs_item=float(w.pvalue) if w is not None else np.nan)
    save_json("e7_summary.json", summ)
    for k, v in summ["span"].items():
        print(k, v)
    print("H6 primary:", summ.get("primary_H6"))
    for k, v in summ["gist"].items():
        print(k, v)


def plot():
    import matplotlib.pyplot as plt
    from hopfield.plotting import INK, save, seq, style
    style()
    d = load_json("e7_raw.json")
    groups = [("M1", "clip"), ("M1", "decay"), ("M4-softmax", "decay"), ("M4-power", "decay")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    ages = np.arange(d["S"])
    for ax, (m, rule) in zip(axes, groups):
        conds = [c for c in d["conds"] if c.startswith(f"{m}|{rule}|")]
        for c, col in zip(conds, seq(len(conds))):
            A = np.array(d["a"][c]).mean(0)
            sm = np.convolve(A, np.ones(9) / 9, mode="same")
            par = c.split("|")[2]
            ax.plot(ages, sm, color=col, label=f"{'A' if rule == 'clip' else 'lambda'} = {par}")
        if m == "M1" and "M1|none|0" in d["a"]:
            A = np.array(d["a"]["M1|none|0"]).mean(0)
            ax.plot(ages, np.convolve(A, np.ones(9) / 9, mode="same"), "--", color="#8c3b2a", label="unbounded")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("memory age (memories stored after it)")
        ax.set_title(f"{m}: {rule}")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("retrieval probability")
    fig.suptitle(f"E7: recall vs age after a stream of S = {d['S']} memories (N = {d['N']})")
    save(fig, f"{FIGURES}/e7_recency.png")

    s = load_json("e7_summary.json")
    nb = d["S"] // d["block"]
    fig, axes = plt.subplots(1, len(d["b_conds"]), figsize=(2.6 * len(d["b_conds"]), 3), sharey=True)
    for ax, c in zip(axes, d["b_conds"]):
        B = np.array(d["b"][c])
        item = B[:, :d["S"]].reshape(len(B), nb, d["block"]).mean(2).mean(0)
        proto = B[:, 2 * d["S"]:].mean(0)
        ax.plot(np.arange(nb), item, "o-", ms=2.5, color="#2f5a40", label="items")
        ax.plot(np.arange(nb), proto, "s-", ms=2.5, color="#c27c2c", label="prototype (gist)")
        ax.set_title(c.replace("|", " "), fontsize=8)
        ax.set_xlabel("block age")
    axes[0].set_ylabel("recall")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"E7b: does the gist outlast the details? (blocks of {d['block']}, c = {d['c']})")
    save(fig, f"{FIGURES}/e7_gist.png")


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
