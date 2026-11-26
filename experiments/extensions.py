"""Optional extensions (proposal section 4).

X1  Binarised MNIST digits as natural correlated memories: E2 with real
    data (memories = digit images, prototype = binarised class mean, never
    stored).
X2  Glauber noise combined with forgetting: E7 (M1, decay) at T > 0.
X3  Key correlations from a small pretrained model (GPT-2) set the key
    similarity of E6; GPT-2 keys are also used directly as the memories of
    one attention head.

Usage: python extensions.py [mnist|glauber|gpt2|all] [--plot-only]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os

import common  # noqa: F401  (must precede numpy: sets BLAS threads = 1)
import numpy as np

from common import FIGURES, N_CUES, N_SETS, QUICK, Timer, load_json, pmap, save_json, warmup
from hopfield.learning import stream_decay
from hopfield.metrics import CATEGORIES, classify, overlap
from hopfield.models import Classical, ContinuousExp, Dense
from hopfield.patterns import flip_bits, random_patterns
from hopfield.seeds import int_seed, rng

# ----------------------------------------------------------------- X1 MNIST
DIGITS = [0, 1, 7]
X1_P = [1, 2, 4, 8, 16, 32, 64, 128, 256]
X1_MODELS = ["M1-Hebb", "M2-n3", "M2-n4", "M2-exp", "M3-b0.02", "M3-b0.1"]
X1_ETA = 0.1
if QUICK:
    X1_P = [1, 8, 64]


def _read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist():
    """Binarised MNIST training set (+-1, threshold 128) and labels."""
    roots = ["/kaggle/input", os.path.join(common.ROOT, "data")]
    imgs = labs = None
    for r in roots:
        cand = [p for p in glob.glob(os.path.join(r, "**", "train-images*idx3*"), recursive=True)
                if os.path.isfile(p)]
        lc = [p for p in glob.glob(os.path.join(r, "**", "train-labels*idx1*"), recursive=True)
              if os.path.isfile(p)]
        if cand and lc:
            imgs, labs = _read_idx(sorted(cand)[0]), _read_idx(sorted(lc)[0])
            break
    if imgs is None:
        import urllib.request
        os.makedirs(os.path.join(common.ROOT, "data"), exist_ok=True)
        base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        for fn in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
            dst = os.path.join(common.ROOT, "data", fn)
            if not os.path.exists(dst):
                urllib.request.urlretrieve(base + fn, dst)
        imgs = _read_idx(os.path.join(common.ROOT, "data", "train-images-idx3-ubyte.gz"))
        labs = _read_idx(os.path.join(common.ROOT, "data", "train-labels-idx1-ubyte.gz"))
    X = np.where(imgs.reshape(len(imgs), -1) >= 128, 1, -1).astype(np.int8)
    return X, labs.astype(int)


def _x1_model(name, xi):
    if name == "M1-Hebb":
        return Classical.hebbian(xi)
    if name.startswith("M2"):
        return Dense(xi, {"M2-n3": 3, "M2-n4": 4, "M2-exp": "exp"}[name])
    return ContinuousExp(xi, float(name.split("-b")[1]))


_MN = None


def task_mnist(args):
    global _MN
    name, s = args
    if _MN is None:
        _MN = load_mnist()
    X, y = _MN
    protos = np.stack([np.where(X[y == dg].mean(0) >= 0, 1, -1) for dg in DIGITS]).astype(np.int8)
    r = rng("X1", s)
    pools = [r.permutation(np.where(y == dg)[0])[:max(X1_P)] for dg in DIGITS]
    out = dict(mem=np.zeros((len(X1_P), 4)), lure=np.zeros((len(X1_P), 4)))
    for ip, p in enumerate(X1_P):
        xi = np.concatenate([X[pl[:p]] for pl in pools])
        labels = np.repeat(np.arange(len(DIGITS)), p)
        m = _x1_model(name, xi)
        cr = rng("X1-cue", s, p)
        targets = cr.choice(len(xi), size=N_CUES, replace=len(xi) < N_CUES)
        lk = cr.integers(0, len(DIGITS), size=N_CUES)
        mc = np.stack([flip_bits(cr, xi[t], X1_ETA) for t in targets])
        lc = np.stack([flip_bits(cr, protos[k], X1_ETA) for k in lk])
        if isinstance(m, ContinuousExp):
            ms, _, _ = m.recall_batch(mc)
            ls, _, _ = m.recall_batch(lc)
        else:
            ms = [m.recall(mc[j], seed=int_seed("X1-dyn", s, p, 0, j)).state for j in range(N_CUES)]
            ls = [m.recall(lc[j], seed=int_seed("X1-dyn", s, p, 1, j)).state for j in range(N_CUES)]
        for j, t in enumerate(targets):
            out["mem"][ip, classify(ms[j], t, xi, protos[labels[t]])[0]] += 1 / N_CUES
        for j, k in enumerate(lk):
            out["lure"][ip, classify(ls[j], None, xi, protos[k])[0]] += 1 / N_CUES
    return out


def run_mnist():
    X, y = load_mnist()
    protos = np.stack([np.where(X[y == dg].mean(0) >= 0, 1, -1) for dg in DIGITS])
    stats = dict(proto_overlaps=overlap(protos, protos),
                 mean_within_class_overlap={str(dg): float(np.mean(overlap(X[y == dg][:200], X[y == dg][:200])))
                                            for dg in DIGITS},
                 mean_image_proto_overlap={str(dg): float(np.mean(overlap(X[y == dg][:500], protos[i])))
                                           for i, dg in enumerate(DIGITS)})
    with Timer("X1 MNIST"):
        t = [(m, s) for m in X1_MODELS for s in range(N_SETS)]
        o = pmap(task_mnist, t)
    data = {}
    for (m, s), v in zip(t, o):
        for k, a in v.items():
            data.setdefault(m, {}).setdefault(k, []).append(a)
    save_json("x1_mnist.json", dict(digits=DIGITS, pgrid=X1_P, models=X1_MODELS, eta=X1_ETA, N=784,
                                    categories=CATEGORIES, stats=stats, data=data))


def plot_mnist():
    import matplotlib.pyplot as plt
    from hopfield.plotting import STATE_COLORS, save, style
    style()
    d = load_json("x1_mnist.json")
    pg = np.array(d["pgrid"])
    fig, axes = plt.subplots(1, len(d["models"]), figsize=(3 * len(d["models"]), 3), sharey=True)
    for ax, m in zip(axes, d["models"]):
        mem = np.array(d["data"][m]["mem"]).mean(0)
        lure = np.array(d["data"][m]["lure"]).mean(0)
        for k, cat in enumerate(d["categories"]):
            ax.plot(pg, mem[:, k], "o-", ms=3, color=STATE_COLORS[cat], label=cat)
        ax.plot(pg, lure[:, 2], "--", color=STATE_COLORS["prototype"], label="lure -> prototype")
        ax.set_xscale("log", base=2)
        ax.set_title(m)
        ax.set_xlabel("images per digit class")
    axes[0].set_ylabel("fraction")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"X1: binarised MNIST digits {d['digits']} as memories (N = 784)")
    save(fig, f"{FIGURES}/x1_mnist.png")


# ------------------------------------------------------- X2 Glauber + forgetting
X2_N, X2_S = 200, 600
X2_LAMS = [0.95, 0.98, 0.99]
X2_TS = [0.0, 0.1, 0.2, 0.3, 0.4]
if QUICK:
    X2_S = 200


def task_glauber(args):
    li, ti, s = args
    xi = random_patterns(rng("E7", s), X2_S, X2_N)  # same streams as E7a
    m = Classical(stream_decay(xi, X2_LAMS[li]))
    cr = rng("E7-cue", s)
    out = np.zeros(X2_S)
    T = X2_TS[ti]
    for mu in range(X2_S):
        r = m.recall(flip_bits(cr, xi[mu], 0.1), seed=int_seed("X2-dyn", s, li, ti, mu), T=T,
                     max_sweeps=50, quench_sweeps=50 if T > 0 else 0)
        out[X2_S - 1 - mu] = overlap(r.state, xi[mu]) >= 0.95
    return out


def run_glauber():
    with Timer("X2 Glauber + forgetting"):
        t = [(li, ti, s) for li in range(len(X2_LAMS)) for ti in range(len(X2_TS)) for s in range(N_SETS)]
        o = pmap(task_glauber, t)
    data = {}
    for (li, ti, s), v in zip(t, o):
        data.setdefault(f"{X2_LAMS[li]}|{X2_TS[ti]}", []).append(v)
    from e7_forgetting import span
    spans = {k: span(np.array(v).mean(0)) for k, v in data.items()}
    save_json("x2_glauber_forgetting.json", dict(N=X2_N, S=X2_S, lams=X2_LAMS, Ts=X2_TS, data=data, span50=spans))
    print(spans)


def plot_glauber():
    import matplotlib.pyplot as plt
    from hopfield.plotting import save, seq, style
    style()
    d = load_json("x2_glauber_forgetting.json")
    fig, axes = plt.subplots(1, len(d["lams"]), figsize=(4.2 * len(d["lams"]), 3.2), sharey=True)
    for ax, lam in zip(axes, d["lams"]):
        for T, col in zip(d["Ts"], seq(len(d["Ts"]))):
            A = np.array(d["data"][f"{lam}|{T}"]).mean(0)
            ax.plot(np.arange(len(A)), np.convolve(A, np.ones(9) / 9, mode="same"), color=col,
                    label=f"T = {T} (span {d['span50'][f'{lam}|{T}']})")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_title(f"M1 decay lambda = {lam}")
        ax.set_xlabel("memory age")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("retrieval (after quench)")
    save(fig, f"{FIGURES}/x2_glauber_forgetting.png")


# ----------------------------------------------------------------- X3 GPT-2
X3_PS = [8, 16, 32, 64, 128, 256, 512, 1000]
X3_ETA_SIGMA = [0.25, 0.5, 1.0]


def gpt2_keys(n_tokens=1024):
    """Keys of every layer/head of GPT-2 small on a public-domain text."""
    import torch
    from transformers import GPT2Model, GPT2TokenizerFast
    text = None
    try:
        import urllib.request
        req = urllib.request.Request("https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                                     headers={"User-Agent": "Mozilla/5.0 (research script)"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        text = raw.decode("utf8", "ignore")
        i = text.find("It is a truth universally acknowledged")
        text = text[i:i + 20000] if i >= 0 else text[5000:25000]
    except Exception as e:  # pragma: no cover
        print("gutenberg download failed:", e)
    if text is None:
        from datasets import load_dataset
        text = " ".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:400])
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()
    ids = tok(text, return_tensors="pt")["input_ids"][:, :n_tokens]
    keys = {}
    hooks = []
    H, dh = model.config.n_head, model.config.n_embd // model.config.n_head
    for li, blk in enumerate(model.h):
        def hook(mod, inp, out, li=li):
            qkv = out[0]  # (T, 3E)
            E = qkv.shape[-1] // 3
            k = qkv[:, E:2 * E].reshape(-1, H, dh).permute(1, 0, 2)
            q = qkv[:, :E].reshape(-1, H, dh).permute(1, 0, 2)
            keys[li] = (q.detach().numpy().copy(), k.detach().numpy().copy())
        hooks.append(blk.attn.c_attn.register_forward_hook(hook))
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return keys, ids.shape[1], dh, text[:200]


def run_gpt2():
    with Timer("X3 GPT-2 keys"):
        keys, T, dh, snippet = gpt2_keys()
        L = len(keys)
        H = keys[0][1].shape[0]
        cos_mean = np.zeros((L, H))
        cos_p90 = np.zeros((L, H))
        cos_nn = np.zeros((L, H))
        r = rng("X3")
        direct = np.zeros((L, H, len(X3_PS), len(X3_ETA_SIGMA)))
        for li in range(L):
            Q, K = keys[li]
            for h in range(H):
                k = K[h]
                kn = k / np.linalg.norm(k, axis=1, keepdims=True)
                C = kn @ kn.T
                iu = np.triu_indices(T, 1)
                cos_mean[li, h] = C[iu].mean()
                cos_p90[li, h] = np.percentile(C[iu], 90)
                np.fill_diagonal(C, -np.inf)
                cos_nn[li, h] = np.mean(C.max(1))
                # direct test: real keys as the memories of one head; random +-1 values;
                # query = target key + Gaussian noise (sigma x key rms); beta = 1/sqrt(d)
                for ip, P in enumerate(X3_PS):
                    if P > T:
                        direct[li, h, ip] = np.nan
                        continue
                    for ie, sg in enumerate(X3_ETA_SIGMA):
                        acc = 0
                        for _ in range(N_CUES):
                            idx = r.choice(T, P, replace=False)
                            Kc = k[idx]
                            V = (2 * r.integers(0, 2, (P, dh)) - 1).astype(float)
                            t = r.integers(P)
                            q = Kc[t] + sg * np.sqrt(np.mean(Kc[t] ** 2)) * r.standard_normal(dh)
                            z = (Kc @ q) / np.sqrt(dh)
                            z -= z.max()
                            p = np.exp(z)
                            p /= p.sum()
                            o = V.T @ p
                            acc += int(np.argmax(V @ o) == t)
                        direct[li, h, ip, ie] = acc / N_CUES
    save_json("x3_gpt2.json", dict(T=T, d=dh, text_snippet=snippet, cos_mean=cos_mean, cos_p90=cos_p90,
                                   cos_nn=cos_nn, Ps=X3_PS, sigmas=X3_ETA_SIGMA, direct=direct))
    print("median key cosine:", float(np.median(cos_mean)), "median nearest-neighbour cosine:",
          float(np.median(cos_nn)))


def plot_gpt2():
    import matplotlib.pyplot as plt
    from hopfield.plotting import save, seq, style
    style()
    d = load_json("x3_gpt2.json")
    cm = np.array(d["cos_mean"])
    nn = np.array(d["cos_nn"])
    D = np.array(d["direct"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    im = axes[0].imshow(cm, aspect="auto", cmap="Greens")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("layer")
    axes[0].set_title("GPT-2: mean pairwise key cosine (= rho)")
    axes[1].hist(cm.ravel(), bins=30, color="#4f7d55", alpha=0.8, label="mean pairwise")
    axes[1].hist(nn.ravel(), bins=30, color="#c27c2c", alpha=0.6, label="nearest neighbour")
    axes[1].set_xlabel("key cosine similarity")
    axes[1].set_ylabel("heads")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Key similarity across the 144 heads")
    for ie, (sg, col) in enumerate(zip(d["sigmas"], seq(len(d["sigmas"])))):
        axes[2].plot(d["Ps"], np.nanmean(D[:, :, :, ie], axis=(0, 1)), "o-", color=col, label=f"query noise {sg}")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("context length P (real GPT-2 keys)")
    axes[2].set_ylabel("retrieval accuracy (mean over heads)")
    axes[2].set_title(r"Real keys as memories, $\beta = 1/\sqrt{d}$")
    axes[2].legend(fontsize=7)
    save(fig, f"{FIGURES}/x3_gpt2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all", choices=["mnist", "glauber", "gpt2", "all"])
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    jobs = ["mnist", "glauber", "gpt2"] if a.which == "all" else [a.which]
    for j in jobs:
        if not a.plot_only:
            if j != "gpt2":
                warmup()
            {"mnist": run_mnist, "glauber": run_glauber, "gpt2": run_gpt2}[j]()
        {"mnist": plot_mnist, "glauber": plot_glauber, "gpt2": plot_gpt2}[j]()
