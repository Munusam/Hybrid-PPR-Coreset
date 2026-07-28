# Hybrid-PPR: Bridging the Gap in Coreset Selection

**Official PyTorch implementation** of *"Bridging the Gap in Coreset Selection: Hybrid Personalized PageRank for Extreme Data Sparsity"*
Under review, CDEL @ ECCV 2026 · Anonymous submission

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-ee4c2c)]()
[![Status](https://img.shields.io/badge/status-under%20review-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Overview

Metric-based coreset selection methods (EL2N, Forgetting Events) rank training
examples by individual difficulty and keep the hardest ones. This works well at
moderate pruning ratios, but at extreme sparsity (≤5% of the data) it backfires:
the coreset fills up almost entirely with outliers, mislabeled samples, and
occluded images, and the network fails to learn even the basic class
prototypes — the **Hard-Example Trap**.

**Hybrid-PPR** addresses this with a two-phase selection strategy:

1. **Prototypical Skeleton** — a uniformly random subset that anchors the
   network on typical, easy-to-learn class examples.
2. **Submodular Boundary Search** — a Personalized PageRank diffusion over a
   k-NN graph of proxy-model embeddings, which propagates EL2N difficulty
   scores through the graph so that only *structurally connected* hard
   regions are up-weighted, while isolated outliers are suppressed. A
   submodular penalty on selected nodes' neighbors keeps the search diverse
   rather than collapsing onto one dense cluster.

```
π^(t+1) = (1 − α) · v + α · Pᵀ · π^(t)
```

where `v` is the L1-normalized EL2N score vector (the restart/personalization
vector), `P` is the row-normalized k-NN adjacency matrix, and `α = 0.15` is
the damping factor.

## Results at a glance

| Setting | Finding |
|---|---|
| Moderate sparsity (50–100%) | Hybrid-PPR is competitive with Random/Herding/Forgetting; no degradation from pruning |
| Extreme sparsity (5%), CIFAR-100 | EL2N/CCS collapse to ~5% accuracy; Hybrid-PPR reaches 12.93% (>150% relative improvement over EL2N) |
| Extreme sparsity (5%), CIFAR-10 | Hybrid-PPR (37.90%) more than doubles EL2N (15.16%) |
| Extreme sparsity (5%), BloodMNIST | Hybrid-PPR (80.30%) outperforms Herding (78.14%) |
| Overall | Distribution-matching baselines (Random, Herding) generally remain the strongest at extreme sparsity — Hybrid-PPR's contribution is rescuing metric-based methods from collapse, not beating every baseline everywhere |

Full per-dataset, per-budget tables are in the paper (Tables 1–2) and in
`results/` after running the experiments below.

> We report this evenhandedly: Hybrid-PPR does **not** universally outperform
> Random/Herding/Forgetting. See the paper's Results section for the complete,
> honest comparison across all methods and budgets.

## Datasets

Automatically downloaded on first run:

| Dataset | Type | Classes | Train size |
|---|---|---|---|
| BloodMNIST | Medical (via `medmnist`) | 8 | 11,959 |
| FashionMNIST | Grayscale | 10 | 60,000 |
| CIFAR-10 | Natural images | 10 | 50,000 |
| CIFAR-100 | Natural images (fine-grained) | 100 | 50,000 |

## Installation

```bash
git clone <repo-url>
cd hybrid-ppr
pip install -r requirements.txt
```

<!-- Confirm requirements.txt exists in the repo before publishing this README.
     At minimum it should pin: torch, torchvision, numpy, pandas, scikit-learn,
     medmnist, matplotlib -->

## Usage

### 1. Run baseline methods (Random, Herding, Forgetting, EL2N, CCS)
The baseline experiments are split into two scripts to manage memory and execution time.
```bash
# Runs baselines
BloodMNIST.ipynb
CIFAR-100_baseline.ipynb
CIFAR-10_baseline.ipynb
Fashion-MNIST_baseline.ipynb

### 2. Run Hybrid-PPR

ppr-blood-MNIST and CIFAR-10.ipynb
ppr-fashion-MNIST and CIFAR-100.ipynb

Both scripts:
- train a lightweight ResNet-18 proxy for 5 warm-up epochs to compute EL2N
  scores and embeddings,
- select a coreset at each of five budgets (100/80/50/25/5%),
- train a fresh ResNet-18 from scratch on each coreset for 30 epochs (Adam,
  lr=1e-3),
- evaluate on the held-out test set and append results to a CSV.

### Method-specific hyperparameters (Hybrid-PPR)

| Parameter | Value | Meaning |
|---|---|---|
| `k_neighbors` | 10 | k-NN graph degree |
| `alpha` | 0.15 | PPR damping factor |
| `gamma` | 0.5 | Fraction of budget from the random skeleton |
| `lambda` | 0.85 | Submodular neighbor-discount penalty |

These were set via informal tuning (see paper, Section on Methodology) rather
than a formal hyperparameter search — noted here for transparency.

## Repository structure

```
.
├── <baseline_script>.py     # Random / Herding / Forgetting / EL2N / CCS
├── <hybrid_ppr_script>.py   # Hybrid-PPR (this paper's method)
├── requirements.txt
└── results/                 # CSV outputs land here after each run
```

<!-- Update this tree to match what's actually committed. -->

## Reproducibility notes

- All results in the paper are **single-run point estimates** (no seed
  averaging yet). Margins ≤0.5% between methods should be read as
  statistically competitive, not a definitive ranking — see the paper's
  "Note on Variance."
- Confirm the ResNet-18 stem modification described in the paper (3×3 first
  conv, no initial MaxPool) matches what's implemented in this repo before
  citing timing/architecture details from the README elsewhere.

## Citation

```bibtex
@inproceedings{anonymous2026hybridppr,
  title     = {Bridging the Gap in Coreset Selection: Hybrid Personalized PageRank for Extreme Data Sparsity},
  author    = {Anonymous Authors},
  booktitle = {CDEL Workshop @ ECCV},
  year      = {2026},
  note      = {Under review}
}
```

## License

MIT — see `LICENSE`.
