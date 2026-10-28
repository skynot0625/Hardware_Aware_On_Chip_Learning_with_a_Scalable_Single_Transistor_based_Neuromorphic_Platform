# Hardware-Aware On-Chip Learning with a Scalable Single-Transistor-based Neuromorphic Platform

PyTorch examples for the manuscript by Sungmin Lee, Jae-Hyun Lee, and colleagues.
The models retain the original research classes and explicit forward computations,
with unused code removed. No Hydra, Lightning, Weights & Biases, or AIHWKit is required.

## Models and datasets

| Model file | Learning modules / architecture | Dataset in this repository |
| --- | --- | --- |
| `models/model_Resnet18_SG.py` | PreActResNet-18, SG | CIFAR-10 |
| `models/model_Resnet_cSG.py` | PreActResNet-18, cSG | CIFAR-10 |
| `models/model_Resnet_SG_eLIFE.py` | PreActResNet-18, SG + eLIFE | CIFAR-10 |
| `models/model_Resenet_cSG_eLIFE.py` | PreActResNet-18, cSG + eLIFE | CIFAR-10 |
| `models/model_Arch1.py` | Recurrent/residual small network, without feedback | SHD |
| `models/model_Arch2.py` | Recurrent/residual small network, with feedback | SHD |

In the paper, the **ResNet family** is evaluated on CIFAR-10, CIFAR-100,
CIFAR10DVS, and DVS128Gesture; the **Arch. 1/Arch. 2 family** is evaluated on
F-MNIST, S-MNIST, and SHD. This release provides **CIFAR-10 and SHD examples only**.
Other input encoders, event-data frontends, and benchmark runners are not included.
Each model file documents its dataset and input shape at the top.

Both small-network files contain SG, cSG, SG+eLIFE, and cSG+eLIFE variants.
Arch1 uses the original no-feedback `_x` classes. Arch2 retains the supplied
four-feedback-path variant (`fb0`–`fb3`). All examples use **SOLO** training.

| Condition | Command-line options |
| --- | --- |
| SG: voltage-domain surrogate | `--surrogate sg` |
| cSG: current-domain surrogate | `--surrogate csg` |
| SG + eLIFE | `--surrogate sg --elife` |
| cSG + eLIFE | `--surrogate csg --elife` |

The surrogate definitions follow main-text Eqs. (5)–(6). eLIFE uses two
recurrently coupled clone neurons with fixed decay factors 0.5 and 1, unit
coupling, and OR eligibility, as in Eqs. (7)–(11) and Supplementary S4.2.

## Installation

Python 3.10 or later; tested with PyTorch 2.5.1 and torchvision 0.20.1.
Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
python verify.py
python verify_conditions.py
```

Run commands from the repository root. CPU is the default; add `--device cuda`
for a compatible CUDA installation.

## CIFAR-10: ResNet-18

CIFAR-10 downloads automatically. Try each condition with a short smoke run:

```bash
python train.py --surrogate sg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg
python train.py --surrogate csg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg
python train.py --surrogate sg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg-elife
python train.py --surrogate csg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg-elife
```

For a full-data run, omit the smoke-run overrides:

```bash
python train.py --surrogate csg --elife --device cuda --output runs/cifar-full
```

Defaults: 5 time steps, width 64, batch size 64, Adam at 1e-4, 300 epochs,
cosine annealing. Width 64 follows the supplied Hydra model config;
`--scale 128` selects the static-CIFAR width reported in Supplementary S4.1.
The original training-wrapper loss is `(0.95 * CE + 0.05 * one_hot_MSE) / 5`.

The official training set is split into 45,000 training and 5,000 validation
images. The official 10,000-image test set is held out. Training uses random
crop, horizontal flip, and normalization; evaluation uses normalization only.

## SHD: Arch1 and Arch2

Download and extract `shd_train.h5` and `shd_test.h5` from the
[official SHD source](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
into `data/SHD/`.

```bash
python train.py --dataset shd --architecture arch1 --data-dir data/SHD --surrogate csg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/arch1
python train.py --dataset shd --architecture arch2 --data-dir data/SHD --surrogate csg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/arch2
```

Use the options in the condition table to run SG or disable eLIFE for either
architecture. SHD defaults to Arch2 when `--architecture` is omitted.

The original models require width **1024**, **20 frames**, and **700 input channels**.
Input shape is `[B,20,700]` or `[B,20,1,700]`; output shape is `[B,20]`.
Defaults: batch size 64, Adam at 1e-3, 100 epochs, cosine annealing, one-hot MSE.
The demo bins event counts into 20 equal-duration bins per sample and reserves
10% of official training samples for validation. This preprocessing is an
explicit example, not a claim to reproduce the original SHD benchmark protocol.

## Outputs and verification

Each run saves its configuration, epoch metrics, validation-selected `best.pt`,
and final test metrics. Choose a new `--output` for each run; existing directories
are not overwritten. `--limit` runs are execution checks, not accuracy benchmarks.

`verify.py` checks SG/cSG and eLIFE gradients. `verify_conditions.py` checks all
12 architecture/condition combinations on CPU and CUDA when available.
The cleanup was additionally checked against the original implementations using
identical retained weights: outputs and retained-parameter gradients matched exactly.

Unused `cTRACE_Node` definitions, intermediate traces, LI nodes, auxiliary
modules, and commented-out experiments were removed. **The eLIFE input trace
is retained because it contributes to the first-layer weight gradient.**
See [source notes](SOURCE_NOTES.md) for the exact adaptation scope and retained
source conventions. Original source hashes are in `SOURCE_PROVENANCE.json`.

## Scope and citation

This is a runnable model release, not a complete reproduction of every paper
figure. Analog-device simulation, continual learning, SPICE, energy estimation,
datasets, manuscript PDFs, and pretrained checkpoints are not distributed.

Please cite *Hardware-Aware On-Chip Learning with a Scalable Single-Transistor-based
Neuromorphic Platform*, Sungmin Lee, Jae-Hyun Lee, and colleagues.
No publication acceptance or DOI is implied. A redistribution license has not
yet been selected; contact the authors regarding reuse.
