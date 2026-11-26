"""Interactive / animated demo of associative recall.

Stores 20x20 letter bitmaps (A, H and optionally the overlapping X) in any
of the four model families, corrupts a cue of 'A' and shows the network
relaxing into an attractor -- the figure on slide 1 of the project deck.

Examples
--------
    python demo/demo.py                         # deck figure for M1, saves demo/out/
    python demo/demo.py --model M2-n3 --eta 0.3 --letters AHX
    python demo/demo.py --model all --gif       # every model, animated GIFs
    python demo/demo.py --show                  # open an interactive window
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hopfield.metrics import binarize, overlap  # noqa: E402
from hopfield.models import Classical, ContinuousExp, Dense, TwoLayer  # noqa: E402
from hopfield.patterns import flip_bits, letter  # noqa: E402

MODELS = ["M1", "M1-Storkey", "M2-n3", "M2-n4", "M2-exp", "M3", "M4-softmax", "M4-power"]
SIDE = 20


def build(name, xi, beta):
    return {"M1": lambda: Classical.hebbian(xi), "M1-Storkey": lambda: Classical.storkey(xi),
            "M2-n3": lambda: Dense(xi, 3), "M2-n4": lambda: Dense(xi, 4), "M2-exp": lambda: Dense(xi, "exp"),
            "M3": lambda: ContinuousExp(xi, beta), "M4-softmax": lambda: TwoLayer(xi, "softmax", beta=beta),
            "M4-power": lambda: TwoLayer(xi, "power", n=4)}[name]()


def frames(model, cue, seed, T):
    """States after every quarter sweep (binary models) or iteration (M3)."""
    N = cue.size
    if isinstance(model, ContinuousExp):
        out = [cue.astype(float)]
        x = cue.astype(float)
        for _ in range(10):
            z = model.beta * (model.X.T @ x)
            z -= z.max()
            p = np.exp(z) / np.exp(z).sum()
            x = model.X @ p
            out.append(x.copy())
        return out
    out = [cue.copy()]
    s = cue.copy()
    for k in range(40):  # 40 quarter sweeps = 10 sweeps
        r = model.recall(s, seed=seed + k, T=T, max_sweeps=1, quench_sweeps=0)
        # one full sweep per call; split visually into quarters by interpolating flips
        changed = np.where(r.state != s)[0]
        for q in range(4):
            part = changed[q * len(changed) // 4:(q + 1) * len(changed) // 4]
            s = s.copy()
            s[part] = r.state[part]
            out.append(s.copy())
        if len(changed) == 0 and T == 0:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="M1", choices=MODELS + ["all"])
    ap.add_argument("--letters", default="AH", help="letters to store (from A, H, X)")
    ap.add_argument("--eta", type=float, default=0.25, help="fraction of cue bits flipped")
    ap.add_argument("--T", type=float, default=0.0, help="Glauber temperature (binary models)")
    ap.add_argument("--beta", type=float, default=0.02, help="inverse temperature (M3, M4-softmax)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gif", action="store_true", help="also save an animated GIF")
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    a = ap.parse_args()
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap

    os.makedirs(a.out, exist_ok=True)
    cmap = ListedColormap(["#dcd9ce", "#2f4b3c"])
    xi = np.stack([letter(c) for c in a.letters])
    rng = np.random.default_rng(a.seed)
    cue = flip_bits(rng, xi[0], a.eta)
    names = MODELS if a.model == "all" else [a.model]

    # deck-style figure: two stored sets ({A,H} and {A,H,X}) side by side
    sets = [a.letters, "AHX" if a.letters == "AH" else a.letters]
    for name in names:
        fig, axes = plt.subplots(3, len(sets), figsize=(3.2 * len(sets), 9.2))
        axes = np.atleast_2d(axes).reshape(3, len(sets))
        anim_frames = None
        for j, letters in enumerate(sets):
            X = np.stack([letter(c) for c in letters])
            m = build(name, X, a.beta)
            fr = frames(m, cue, a.seed, a.T)
            mid = fr[min(1, len(fr) - 1)]
            for i, (st, lab) in enumerate([(cue, f"cue: {int(a.eta * 100)}% of bits flipped"),
                                           (mid, "relaxing (1/4 sweep)"), (fr[-1], "final state")]):
                ax = axes[i, j]
                ax.imshow(np.asarray(binarize(st) if name == "M3" else st).reshape(SIDE, SIDE), cmap=cmap,
                          vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"overlap with A: m = {overlap(st, X[0]):.2f}")
                if j == 0:
                    ax.set_ylabel(lab)
            axes[0, j].set_title(f"{len(letters)} stored memories\n{{{', '.join(letters)}}}")
            if j == len(sets) - 1:
                anim_frames = (fr, X)
        fig.suptitle(f"{name}: pattern completion from a corrupted cue")
        fig.savefig(os.path.join(a.out, f"recall_{name}.png"), dpi=130, bbox_inches="tight")
        if a.gif:
            fr, X = anim_frames
            fig2, ax2 = plt.subplots(figsize=(3, 3.3))
            im = ax2.imshow(fr[0].reshape(SIDE, SIDE), cmap=cmap, vmin=-1, vmax=1)
            ax2.set_xticks([])
            ax2.set_yticks([])

            def upd(k):
                st = fr[k]
                im.set_data(np.asarray(binarize(st) if name == "M3" else st).reshape(SIDE, SIDE))
                ax2.set_title(f"{name}  step {k}  m(A) = {overlap(st, X[0]):.2f}", fontsize=8)
                return [im]
            FuncAnimation(fig2, upd, frames=len(fr), interval=150).save(
                os.path.join(a.out, f"recall_{name}.gif"), writer=PillowWriter(fps=6))
            plt.close(fig2)
        if a.show:
            plt.show()
        plt.close(fig)
        print("saved", os.path.join(a.out, f"recall_{name}.png"))


if __name__ == "__main__":
    main()
