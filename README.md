This is the **official PyTorch implementation** of our NeurIPS 2023 paper:

> **Data Pruning via Moving-one-Sample-out**
> Haoru Tan*, Sitong Wu*, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi
> *The University of Hong Kong, The Chinese University of Hong Kong, DAMO Academy (Alibaba Group), Hupan Lab*
> NeurIPS 2023 &nbsp;|&nbsp; [[Paper (arXiv:2310.14664)]](https://arxiv.org/abs/2310.14664)

---

## Overview

Modern deep learning relies on ever-larger datasets, many of which contain **redundant or noisy** samples. **MoSo** is a data-pruning method that ranks each training sample by **how much it would change the optimal empirical risk if it were removed from the training set**:

$$
\mathcal{M}(z) \;=\; \mathcal{L}\!\big(S\setminus z,\, w^{\*}_{S\setminus z}\big) \;-\; \mathcal{L}\!\big(S\setminus z,\, w^{\*}_{S}\big).
$$

Computing this exactly via leave-one-out retraining is hopeless (≈45 GPU-years on ImageNet-1K). We instead derive a **first-order, training-dynamics-aware estimator** with linear complexity and bounded approximation error:

$$
\widehat{\mathcal{M}}(z) \;=\; \mathbb{E}_{t\sim \mathcal{U}\{1,\dots,T\}}\!\Big[\tfrac{T}{N}\,\eta_t\, \nabla \mathcal{L}(S\setminus z,\, w_t)^{\top}\,\nabla \ell(z,\, w_t)\Big].
$$

Intuition: a sample whose gradient consistently aligns with the **average** gradient over the whole training trajectory is informative, and gets a high score. Noisy samples and outliers receive *low* (or negative) scores and are pruned first.


## Repository structure

```
MoSo/
├── surrogate_training.py   # Stage 1: train surrogate network(s) with dataset partitioning
├── scoring.py              # Stage 2: compute MoSo scores from saved checkpoints
├── retraining.py           # Stage 3: retrain target network on the MoSo-pruned subset
├── models/                 # Standard CIFAR-style backbones (ResNet, SENet, EfficientNet, ...)
├── model.py                # Auxiliary model wrappers
├── utils.py                # Progress bar and small helpers
├── Tiny_preprocessing.sh   # Re-organize Tiny-ImageNet val/ into class folders
└── main.py                 # Legacy single-file pipeline (kept for reference)
```

The recommended entry points are the three numbered stages (`surrogate_training.py`, `scoring.py`, `retraining.py`). `main.py` is the older monolithic script and is kept only for backward compatibility.

---

## Installation

The code requires Python 3.8+ and PyTorch 1.10+ (any recent version with `torchvision` and CUDA should work).

```bash
git clone https://github.com/hrtan/MoSo.git
cd MoSo

# Recommended: create a fresh environment
conda create -n moso python=3.9 -y
conda activate moso

# Install PyTorch matching your CUDA version (see https://pytorch.org/)
pip install torch torchvision

pip install numpy tqdm
```

---

## Datasets

### CIFAR-10 / CIFAR-100

Both will be downloaded automatically by `torchvision`. By default the code looks for the data under hard-coded paths inherited from our internal cluster — please **edit the `root=...` arguments** in the three stage scripts to point to a directory you can write to, e.g.:

```python
trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, ...)
```

### Tiny-ImageNet

Download Tiny-ImageNet-200 from the [official source](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it. The validation split needs to be re-organized into class sub-folders before it can be loaded by `ImageFolder`. Edit the `current=` path at the top of `Tiny_preprocessing.sh` and run:

```bash
bash Tiny_preprocessing.sh
```

Then update the `train_set_path` / `test_set_path` at the top of each stage script to the location of your `tiny-imagenet-200` directory.

### ImageNet-1K

Use the standard `train/` and `val/` ImageFolder layout. The pipeline is the same as for the smaller datasets; just plug in an `ImageFolder` instead of `CIFAR{10,100}`.

---

## Quick start (CIFAR-100 with ResNet-50)

The whole pipeline boils down to three commands. All artifacts (checkpoints, MoSo scores, retrained models) live under `--path`, so use the **same `--path` value** across the three stages.

### Stage 1 — Train the surrogate network(s)

We follow the parallel-acceleration scheme from the paper: the training set `S` is split into `num_trails` non-overlapping subsets `{S_1, ..., S_I}`, and a small surrogate network is trained on each. Checkpoints from every epoch are saved to `<path>/checkpoint/trial_<i>_<epoch>.pth`.

```bash
python surrogate_training.py \
    --dataset cifar100 \
    --model   resnet50 \
    --bs      256 \
    --lr      0.1 \
    --maxepoch  50 \
    --num_trails 8 \
    --path    ./MoSo_CIFAR100
```

> **Tip.** Larger `--num_trails` makes a single sample's contribution easier to detect (see Table 1 of the paper), but reduces the size of each surrogate training set. We use `num_trails=8` for CIFAR-100 and Tiny-ImageNet.

### Stage 2 — Compute MoSo scores

Sample `--samples` checkpoints uniformly along training (this estimates the expectation in Eq. 4) and accumulate per-sample scores into `<path>/score/moso_score.pth`.

```bash
python scoring.py \
    --dataset cifar100 \
    --model   resnet50 \
    --bs      1 \
    --maxepoch  50 \
    --num_trails 8 \
    --samples 10 \
    --path    ./MoSo_CIFAR100
```

### Stage 3 — Retrain on the MoSo-pruned coreset

Load the scores from Stage 2, keep the top-`(1 - pr)` fraction (class-balanced), and retrain a fresh network from scratch:

```bash
python retraining.py \
    --dataset cifar100 \
    --model   resnet50 \
    --pr      0.5 \
    --bs      256 \
    --lr      0.1 \
    --maxepoch  200 \
    --num_trails 8 \
    --path    ./MoSo_CIFAR100
```

`--pr` is the **pruning ratio** (`0.5` keeps half the data, `0.8` keeps 20%). Set `--random 1` to retrain on a randomly pruned subset of the same size — useful as a sanity-check baseline.

---

## Reproducing the paper experiments

| Experiment | Dataset | Surrogate model | Target model | Stage 3 args |
|---|---|---|---|---|
| Main pruning curves (Fig. 1a) | CIFAR-100 | ResNet-50 | ResNet-50 | `--pr {0.2, 0.4, 0.6, 0.7, 0.8}` |
| Main pruning curves (Fig. 1b) | Tiny-ImageNet | ResNet-50 | ResNet-50 | `--dataset tiny --pr ...` |
| Generalization to SENet (Fig. 3a) | CIFAR-100 | ResNet-50 | SENet | `--model senet` in Stage 3 |
| Generalization to EfficientNet (Fig. 3b) | CIFAR-100 | ResNet-50 | EfficientNet-B0 | `--model EfficientNetB0` |
| Robustness to label noise (Fig. 3c/d) | CIFAR-100 | ResNet-50 | ResNet-50 | `--noise_ratio 0.2` in **all three** stages |

Notes:
- For the **noisy-label** experiments, generate a noise mask once and place it under `<path>/noise_mask/label.pth` (a `torch.long` tensor of length `len(trainset)` with the corrupted labels). The stage scripts will pick it up automatically when `--noise_ratio > 0`.
- For the **architecture-transfer** experiments, you only need to re-run **Stage 3** with a different `--model`; the MoSo scores produced in Stage 2 are reused as-is.

---

## Key arguments (cheat sheet)

| Flag | Used in | Meaning |
|---|---|---|
| `--dataset` | all stages | `cifar10` / `cifar100` / `tiny` |
| `--model` | all stages | Backbone for surrogate/target. Supports `resnet18`, `resnet50`, `senet`, `mobilenetv2`, `EfficientNetB0` (Stage 3 only) |
| `--path` | all stages | Experiment root; reused across stages |
| `--num_trails` | Stages 1 & 2 | Number of dataset partitions / parallel surrogates (the `I` in Algorithm 1) |
| `--maxepoch` | Stages 1 & 3 | 50 epochs is usually enough for the surrogate (Table 2); 200 is the standard for retraining |
| `--samples` | Stage 2 | Number of timesteps sampled to estimate the expectation in Eq. 4 |
| `--pr` | Stage 3 | Pruning ratio; e.g. `0.8` means 80% of data is discarded |
| `--noise_ratio` | all stages | Synthetic-label-noise rate (0.0 by default) |
| `--trainaug` | Stages 1 & 3 | `0`: none, `1`: AutoAugment (CIFAR-10 only), `2`: RandAugment, `3`: AugMix |
| `--random` | Stage 3 | `1` enables random pruning (baseline) |

---

## Algorithm at a glance

```
Input : dataset S, pruning ratio δ, surrogate iterations T, partitions I
Output: pruned coreset Ŝ ⊂ S of size (1-δ)|S|

1.  Partition S into S_1, ..., S_I    (one per GPU)
2.  for each S_i in parallel:
       train surrogate net on S_i for T epochs, saving { (w_t, η_t) }
3.  for each sample z in S_i:
       M̂(z) = E_t [ η_t · ⟨ ∇L(S_i\z, w_t),  ∇ℓ(z, w_t) ⟩ ]
4.  Combine M̂(·) across partitions → global score vector
5.  Ŝ ← top-(1-δ) samples by M̂ (class-balanced)
```

This is exactly Algorithm 1 of the paper; the three Python files implement steps 1–2, 3–4, and the downstream training on `Ŝ`, respectively.

---

## Citation

If you find MoSo useful in your research, please cite:

```bibtex
@inproceedings{tan2023moso,
  title     = {Data Pruning via Moving-one-Sample-out},
  author    = {Tan, Haoru and Wu, Sitong and Du, Fei and Chen, Yukang and
               Wang, Zhibin and Wang, Fan and Qi, Xiaojuan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}
```

