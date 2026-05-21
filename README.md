This is the **official PyTorch implementation** of our NeurIPS 2023 paper:

> **Data Pruning via Moving-one-Sample-out**
> Haoru Tan*, Sitong Wu*, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi
> *The University of Hong Kong, The Chinese University of Hong Kong, DAMO Academy (Alibaba Group), Hupan Lab*
> NeurIPS 2023 &nbsp;|&nbsp; [[Paper (arXiv:2310.14664)]](https://arxiv.org/abs/2310.14664)




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

