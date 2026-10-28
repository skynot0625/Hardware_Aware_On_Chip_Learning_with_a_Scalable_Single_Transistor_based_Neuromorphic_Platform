# Hardware-Aware On-Chip Learning with a Scalable Single-Transistor-based Neuromorphic Platform

Standalone PyTorch model examples accompanying the manuscript by Sungmin Lee,
Jae-Hyun Lee, and colleagues. These examples expose the learning modules without
Hydra, Lightning, Weights & Biases, or AIHWKit.

## Scope

Both the pre-activation ResNet-18 and digital SHD feedback network support:

| Condition | Gradient window | Eligibility | CLI |
| --- | --- | --- | --- |
| SG | Membrane potential | Disabled | `--surrogate sg` |
| cSG | Instantaneous synaptic current | Disabled | `--surrogate csg` |
| SG + eLIFE | Membrane potential | Enabled | `--surrogate sg --elife` |
| cSG + eLIFE | Instantaneous synaptic current | Enabled | `--surrogate csg --elife` |

All four conditions use **SOLO**: forward simulation across the sequence and
spatial gradients at the final selected step only. SG does not imply BPTT.
SG and cSG share the same hard-spike forward dynamics; only the backward mask
differs. Both windows have center 1 and half-width 1 for this matched example.

These are runnable algorithm examples, **not a complete reproduction of all
figures or reported accuracies**. Device measurements, analog nonidealities,
continual-learning protocols, SPICE, and energy estimation are not included.
No pretrained checkpoints or manuscript PDFs are distributed here.

## Installation

Python 3.10 or later. Tested runtime: PyTorch 2.5.1 and torchvision 0.20.1.
Create an isolated environment and install `requirements.txt` with your package
manager. For example, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
python verify.py
```

CPU is the default. Pass `--device cuda` for a compatible CUDA installation.
Dataset downloads require Internet access and disk space; training outputs are
excluded from Git.

## Quick start: CIFAR-10, all four conditions

CIFAR-10 is downloaded automatically on first use. These short runs use only
16 samples per split and reduced width; they test execution, not accuracy.

```bash
python train.py --surrogate sg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg
python train.py --surrogate csg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg
python train.py --surrogate sg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg-elife
python train.py --surrogate csg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg-elife
```

For a full-data run, remove `--limit` and the reduced-width/epoch overrides:

```bash
python train.py --surrogate csg --elife --device cuda --output runs/cifar-full
```

Defaults: 5 time steps, width 64, batch size 64, Adam at 1e-4,
300 epochs, cosine annealing. Width 64 follows the supplied Hydra model config;
Supplementary S4.1 reports width 128 for static CIFAR experiments, selectable
with `--scale 128`. The standalone schedule follows Supplementary S4.1 rather
than the repository's generic experiment override (`lr=0.01`). The active
source wrapper loss is retained: `(0.95 * cross_entropy + 0.05 * MSE) / 5`.

The official 50,000-image training split is divided into 45,000 training and
5,000 validation images with a seeded permutation. The official 10,000-image
test split remains untouched. Only training images receive crop/flip augmentation.
The original template pooled train and test before repartitioning; this example
intentionally does not copy that behavior. Cutout is not included.

## Digital SHD small network

Download and extract `shd_train.h5` and `shd_test.h5` from the
[official SHD source](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
into `data/SHD/`. No AIHWKit installation is required.

```bash
python train.py --dataset shd --data-dir data/SHD --surrogate sg --epochs 1 --scale 32 --limit 16 --output runs/shd-sg
python train.py --dataset shd --data-dir data/SHD --surrogate csg --epochs 1 --scale 32 --limit 16 --output runs/shd-csg
python train.py --dataset shd --data-dir data/SHD --surrogate sg --elife --epochs 1 --scale 32 --limit 16 --output runs/shd-sg-elife
python train.py --dataset shd --data-dir data/SHD --surrogate csg --elife --epochs 1 --scale 32 --limit 16 --output runs/shd-csg-elife
```

Defaults: width 1024, 20 frames, batch size 64, Adam at 1e-3, 100 epochs,
cosine annealing, one-hot MSE. Training selects a random prefix of 4–20 frames;
evaluation uses all 20. Input shape is `[batch,20,700]` or `[batch,20,1,700]`.
Twenty output classes use 200 readout units averaged in groups of ten.
The network preserves two recurrent paths, residual projections and four
feedback paths; all synapses are bias-free `nn.Linear` layers.

This demo bins event counts into 20 equal-duration bins per sample and reserves
10% of official training samples for validation. The original preprocessing
pipeline was not supplied with the model; this choice is not claimed to
reproduce the paper's SHD scores or its Classes 0–9 adaptation protocol.

## Implementation notes and provenance

- `models/model_current.py` and `models/model_current_eLIFE.py` expose the SOLO
  variants extracted from the corresponding original files. Shared ResNet code
  lives in `models/resnet.py`; state-dict names differ from the original.
- `models/shd.py` adapts `model_analog.py`'s
  `mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3`. Analog layers are replaced
  by digital layers, using Xavier initialization from its digital `_pure` variant.
  The source loop skipped the penultimate frame; this example processes every
  frame of the selected prefix. Unused modules were omitted.
- `models/neurons.py` implements main-text Eqs. (3)–(11) and Supplementary S4.2:
  fixed clone decay 0.5/1, unit input/cross-coupling, delayed simultaneous updates,
  and logical OR eligibility. No sigmoid is applied to fixed clone coefficients.
  The original ResNet clone's membrane overwrite is removed.
- Firing uses strict `V > 1`; gradient windows use strict `abs(signal-1) < 1`,
  preserving the source's boundary convention. Eligibility changes the
  weight-gradient path, not forward synaptic values. OR passes gradients only
  to main spikes, not clone states.
- The digital input projection retains the original continuous input trace when
  eLIFE is enabled. Thus this is a software demonstration of the learning modules,
  not an entirely state-access-free hardware implementation of every operation.
- Original effective initialization is retained: main membrane **and initial
  spike** tensors start at 0.5 due to source aliasing. SHD main-neuron decay logits
  start at zero (effective decay 0.5); two stages have per-neuron trainable decay.
  These source choices should not be confused with the fixed clone coefficients.

## Outputs and verification

Each run creates `config.json`, `metrics.jsonl`, validation-selected `best.pt`,
and `test.json`. Existing output directories are rejected to prevent overwrites.
Test evaluation occurs after checkpoint selection, never for model selection.
Seeds are recorded; bitwise reproducibility across hardware is not guaranteed.

`python verify.py` checks cSG gradients, clone dynamics, eligibility gradients,
state reset and optimizer updates. `python verify_conditions.py` checks all four
conditions on both architectures, including SG/cSG forward equality and
different backward masks. These checks are not benchmark training.

## Citation and availability

Please cite the manuscript **Hardware-Aware On-Chip Learning with a Scalable
Single-Transistor-based Neuromorphic Platform**, Sungmin Lee, Jae-Hyun Lee,
and colleagues. Publication metadata will be added when available; no DOI or
acceptance status is implied. The provided manuscript's Code availability text
describes release upon acceptance and should be updated if this example is
released publicly during review.

No license has been selected yet. Public visibility alone does not grant an
open-source redistribution license; contact the authors regarding reuse.
