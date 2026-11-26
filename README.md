# From Memorization to Generalization

**Interference, false memories, forgetting and robustness in classical and modern Hopfield networks.**
INCM 2026 · Project 15 · Athmeeya M Kashyap, Laksh Mittal

This repository holds a from-scratch simulator (NumPy and Numba, no Hopfield libraries) for four
families of associative memory. It runs every experiment of the proposal (Part A, E1–E7 and the
optional extensions) and contains the results, the figures, the report and a demo.

**Report:** [report/report.pdf](report/report.pdf) (source: [report/report.tex](report/report.tex))

## Models
| | Model | Energy / update |
|---|---|---|
| M1 | Classical (Hebbian or Storkey) | `E = -½ Σ W_ij s_i s_j` |
| M2 | Dense associative memory | `E = -Σ_μ F(ξ^μ·s)`, `F = x^n` (n = 2, 3, 4) or `e^x` (log-space) |
| M3 | Continuous exponential (= one attention head) | `x ← X softmax(β Xᵀx)` |
| M4 | Two-layer feature/memory-neuron circuit | `v_i ← sgn Σ_μ W_iμ g_μ F'(h_μ)` or softmax; memory neurons can be lesioned or decay |

The variations implemented are Glauber dynamics with temperature T, prototype-structured memories with
similarity c, and forgetting by clipping (Parisi) or decay (Mézard–Nadal–Toulouse; in M4, decaying
memory-neuron strengths).

## Results (50 pattern sets × 20 cues per condition)
| | Primary test | Result | Holm p |
|---|---|---|---|
| H1 | capacity slope, M2 n=4 minus M1 | 2.27 [2.23, 2.30] (n=3: 1.92, n=4: 3.08; exp: e^{0.34N}) | 0.001 ✔ |
| H2 | P\*(n=3) / P\*(M1), c = 0.4 | 3.5× (P\* = 8, 29, 131 for n = 2, 3, 4; never for exp) | 0.001 ✔ |
| H3 | single-recall vs β in M3 | Spearman 0.88; three regimes: single / prototype / blend | 1e-10 ✔ |
| H4 | all-or-none index, M4 neuron vs M1 synapse lesion | 1.00 vs 0.68 | 3e-9 ✔ |
| H5 | attention head, key similarity 0 vs 0.8 | accuracy 1.00 vs 0.43 | 3e-9 ✔ |
| H6 | recent-20 recall, decay vs unbounded M1 | 0.47 vs 0.00 (span ≈ 0.05N) | 3e-9 ✔ |

Some secondary predictions were not borne out, and the report says so:
- Moderate noise did not help escape wrong attractors.
- The prototype did not outlast items under memory-neuron deletion or decay in M4.
- The Hebbian α_c extrapolation (0.133 in N^-½, 0.152 in N^-1) brackets 0.138 but does not tightly confirm it.

## Layout
```
hopfield/        simulator: patterns, learning rules, Numba kernels, models, metrics, statistics, plotting
experiments/     partA.py, e1_capacity.py ... e7_forgetting.py, extensions.py (MNIST, Glauber+forgetting, GPT-2),
                 hypothesis_tests.py (primary tests + Holm), collect.py (merges per-job outputs)
tests/           correctness checks: monotone energy, fixed points, M2(n=2) == M1 exactly, Storkey reference, ...
results/         raw and summary JSON/NPZ for every experiment; primary_tests.json
figures/         all figures (PNG + PDF)
demo/            demo.py and rendered output (demo/out: PNG + animated GIF per model)
report/          report.tex / report.pdf
```

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_all.sh              # full run: tests, Part A, E1-E7, extensions, primary tests
QUICK=1 ./run_all.sh      # smoke test (4 sets x 5 cues)
cd experiments && python e2_generalization.py --plot-only   # redraw one experiment's figures
```
Every random quantity derives from one master seed (`hopfield/seeds.py`) through `numpy.random.SeedSequence`,
addressed by experiment, condition and pattern set. All models see identical patterns, cues and update orders,
so model comparisons are paired. The X3 extension needs `torch` and `transformers`, and downloads GPT-2 and a
public-domain text.

The reported results were computed on Kaggle CPU kernels: one kernel per experiment, then a collect kernel
that merged the outputs and ran the primary tests.

## Demo
```bash
python demo/demo.py                       # A/H/X letter recall (deck slide 1), classical network
python demo/demo.py --model all --gif     # every model, with animations
python demo/demo.py --model M2-n3 --eta 0.35 --letters AHX --show
```
