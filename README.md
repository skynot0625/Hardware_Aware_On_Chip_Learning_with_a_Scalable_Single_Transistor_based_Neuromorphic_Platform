# Hardware-Aware On-Chip Learning with a Scalable Single-Transistor-based Neuromorphic Platform

Source-preserving PyTorch examples accompanying the manuscript by Sungmin Lee,
Jae-Hyun Lee, and colleagues. The model files retain the original research
classes, layer names, initialization, and explicit forward code. Hydra,
Lightning, Weights & Biases and AIHWKit are not needed to run these examples.

## Models and four learning conditions

| Condition | ResNet source file | CLI |
| --- | --- | --- |
| SG | `models/model.py` | `--surrogate sg` |
| cSG | `models/model_current.py` | `--surrogate csg` |
| SG + eLIFE | `models/model_eLIFE.py` | `--surrogate sg --elife` |
| cSG + eLIFE | `models/model_current_eLIFE.py` | `--surrogate csg --elife` |

The first, second and fourth files contain the original effective
`SOLO_PreactResNet18` class, rather than a rewritten shared architecture.
SG+eLIFE is a minimal derivative of the original cSG+eLIFE file: the main
neurons use `HeavisideBoxcarCall` and pass membrane potential alone to it.
The forward architecture and eLIFE operations are unchanged.

`models/model_shd.py` contains these original small-network classes:

| Condition | Original class |
| --- | --- |
| SG | `mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_pure` |
| cSG | `mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_pure` |
| SG + eLIFE | `mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3` |
| cSG + eLIFE | `mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3` |

All use SOLO, not BPTT. SG computes a voltage-domain boxcar; cSG computes a
current-domain boxcar (main-text Eqs. 5–6). The example passes center 1 and
half-width 1 to both. eLIFE uses two clones with fixed decay 0.5/1 and unit
coupling, followed by OR eligibility (Eqs. 7–11; Supplementary S4.2).
Firing uses the original strict `>` boundary convention.

## Installation

Python 3.10 or later. Tested with PyTorch 2.5.1 and torchvision 0.20.1.
For example, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
python verify.py
python verify_conditions.py
```

CPU is the default. Add `--device cuda` to training commands for CUDA.

## Quick start: CIFAR-10

The dataset downloads automatically. Each command below runs one short
training/validation/test cycle using 16 samples per split and reduced width.
These are execution checks, not paper accuracy benchmarks.

```bash
python train.py --surrogate sg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg
python train.py --surrogate csg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg
python train.py --surrogate sg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg-elife
python train.py --surrogate csg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg-elife
```

For full-data training:

```bash
python train.py --surrogate csg --elife --device cuda --output runs/cifar-full
```

Defaults: width 64 and 5 steps from the Hydra model configuration, batch size
64, Adam at 1e-4, 300 epochs and cosine annealing. Supplementary S4.1 reports
width 128 for CIFAR; use `--scale 128` for that width. The schedule follows
Supplementary S4.1 rather than the generic experiment override (`lr=0.01`).
The active original Lightning loss is retained:
`(0.95 * cross_entropy + 0.05 * one_hot_MSE) / 5`.

The standalone loader reserves 5,000 official training images for validation
and leaves the official test set untouched. This intentionally differs from
the template loader that pooled training and test before repartitioning.
Training uses random crop, horizontal flip and source normalization; evaluation
uses normalization only. Cutout is not included.

## SHD: original small network with digital layers

Download and extract `shd_train.h5` and `shd_test.h5` from the
[official source](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
into `data/SHD/`.

```bash
python train.py --dataset shd --data-dir data/SHD --surrogate sg --epochs 1 --batch-size 2 --limit 8 --output runs/shd-sg
python train.py --dataset shd --data-dir data/SHD --surrogate csg --epochs 1 --batch-size 2 --limit 8 --output runs/shd-csg
python train.py --dataset shd --data-dir data/SHD --surrogate sg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/shd-sg-elife
python train.py --dataset shd --data-dir data/SHD --surrogate csg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/shd-csg-elife
```

The source hard-codes width **1024** and **20 frames**; these are preserved.
The CLI rejects other SHD widths rather than silently rewriting the network.
Inputs are `[B,20,700]` or `[B,20,1,700]`; outputs are 20 class scores obtained
by averaging groups of ten readout units. Defaults are Adam at 1e-3, 100 epochs,
batch size 64, cosine annealing and one-hot MSE.

The demo bins event counts into 20 equal-duration bins per sample and reserves
10% of training samples for validation. The original SHD preprocessing was not
provided with the model, so this loader is not claimed to reproduce paper
accuracy or the separate Classes 0–9 continual-learning protocol.

## Exactly what was changed in the model sources

`export_originals.py` extracts class text directly, without AST unparse or
architecture refactoring. When a source defines a class more than once, the
last definition is retained, matching Python's effective definition. Unused
architectures/imports and import-time random seeding are omitted. Seeding is
instead performed by `train.py`.

The complete mechanical model changes are:

1. Use local `surrogate` imports and remove unused template dependencies.
2. Remove automatic device placement from constructors; the caller selects
   CPU/CUDA with `model.to(device)`.
3. Remove the clone `LIF_U = I_in` overwrite in the ResNet eLIFE helper, as
   requested, retaining the manuscript's leak/integration equation.
4. Replace SHD `AnalogLinear` calls with `nn.Linear` and remove their
   `rpu_config` argument. Original initialization and unused modules remain.
5. Correct `surrogate.eLIF.backward` to return two gradients for its two
   inputs, rather than three. Forward behavior is unchanged.
6. Derive ResNet SG+eLIFE by replacing the main current-surrogate call with the
   voltage-surrogate call. This is a derived variant, not a separately located
   original source file.

Original source filenames and SHA-256 hashes are recorded in
`SOURCE_PROVENANCE.json`. To reproduce the exports from an original checkout:

```bash
python export_originals.py --solo-dir /path/to/ml/src/SOLO --analog-model /path/to/model_analog.py
```

**Original quirks are preserved, not silently corrected:** the SHD prefix loop
uses `range(init_step, last_step - 1)` and therefore skips the penultimate
frame; aliased initial main membrane/spike tensors are both filled with 0.5;
SHD main-neuron decay logits are overwritten to zero in the source helpers;
the original variants have different initialization routines. Consequently,
these examples should not be interpreted as a newly controlled ablation or
used to claim exact reproduction without checking the original experiment
protocol. The eLIFE-enabled input projection also retains a continuous input
trace, so the software implementation is not itself a complete hardware system.

## Outputs, verification and scope

Each run saves `config.json`, `metrics.jsonl`, validation-selected `best.pt`
and final `test.json`. Existing output directories are rejected. Model
selection does not use the test set. Seeds are recorded; cross-device bitwise
reproducibility is not guaranteed.

`verify.py` checks original SG/cSG and eLIFE equations/gradients.
`verify_conditions.py` runs all four conditions for both original networks,
checking state reset, finite gradients and optimizer updates on CPU and CUDA
when available. Original layer names and model parameter keys are retained;
the initial compact release's checkpoints are not compatible with this version.

This repository provides runnable model examples, **not all paper results**.
Device measurements, analog nonidealities, SPICE, continual learning and energy
estimation are outside its scope. PDFs, datasets and checkpoints are not
distributed. No publication acceptance or DOI is implied.

Please cite **Hardware-Aware On-Chip Learning with a Scalable
Single-Transistor-based Neuromorphic Platform**, Sungmin Lee, Jae-Hyun Lee,
and colleagues. The manuscript's Code availability statement should reflect
the actual scope of this public example release.

No redistribution license has yet been selected; contact the authors regarding reuse.
