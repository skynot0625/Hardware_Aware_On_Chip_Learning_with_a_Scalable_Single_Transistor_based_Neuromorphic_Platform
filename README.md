# Hardware-Aware On-Chip Learning with a Scalable Single-Transistor-based Neuromorphic Platform

Algorithm-level examples of the **current-based surrogate gradient (cSG)** and
**eligible leaky integrate-and-fire encoder (eLIFE)** described in the manuscript.

cSG derives the surrogate signal from instantaneous synaptic current rather
than membrane potential. eLIFE uses two recurrently connected clone neurons
to generate spike-based eligibility signals. These modules are intended to
reduce the need for external access to internal neuronal states during learning.
See main-text Methods, Eqs. (5)–(11), and Supplementary Note S4.2.

## Models

| File | Architecture / learning modules | Example dataset |
| --- | --- | --- |
| `models/model_Arch1.py` | Arch. 1: recurrent network without feedback | SHD |
| `models/model_Arch2.py` | Arch. 2: recurrent network with feedback | SHD |
| `models/model_ResNet18_SG.py` | PreActResNet-18, SG | CIFAR-10 |
| `models/model_ResNet18_cSG.py` | PreActResNet-18, cSG | CIFAR-10 |
| `models/model_ResNet18_SG_eLIFE.py` | PreActResNet-18, SG + eLIFE | CIFAR-10 |
| `models/model_ResNet18_cSG_eLIFE.py` | PreActResNet-18, cSG + eLIFE | CIFAR-10 |

Arch1 and Arch2 each provide four variants, named `Arch1_SG`, `Arch1_cSG`,
`Arch1_SG_eLIFE`, `Arch1_cSG_eLIFE`, and correspondingly for Arch2.
The examples use SOLO, with spatial gradients evaluated at the final selected
time step. `eP` denotes **eligible potential** in the digital input-learning path.

The main implementations are in `models/surrogate.py` (SG/cSG),
`models/shd_nodes.py` (small-network neuron and eLIFE helpers), and the model files.

## Setup

Python 3.10 or later; PyTorch 2.5.1 and torchvision 0.20.1.
For example, using [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
source .venv/bin/activate
```

Run commands from the repository root. CPU is the default; use `--device cuda`
for a compatible CUDA installation.

## Run an example

Choose the learning condition with:

| Condition | Options |
| --- | --- |
| SG | `--surrogate sg` |
| cSG | `--surrogate csg` |
| SG + eLIFE | `--surrogate sg --elife` |
| cSG + eLIFE | `--surrogate csg --elife` |

### SHD

Download and extract `shd_train.h5` and `shd_test.h5` from the
[official SHD source](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
into `data/SHD/`.

```bash
python train.py --dataset shd --architecture arch1 --data-dir data/SHD --surrogate csg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/arch1
python train.py --dataset shd --architecture arch2 --data-dir data/SHD --surrogate csg --elife --epochs 1 --batch-size 2 --limit 8 --output runs/arch2
```

### CIFAR-10

CIFAR-10 downloads automatically on first use.

```bash
python train.py --dataset cifar10 --surrogate sg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg
python train.py --dataset cifar10 --surrogate csg --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg
python train.py --dataset cifar10 --surrogate sg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/sg-elife
python train.py --dataset cifar10 --surrogate csg --elife --epochs 1 --scale 4 --batch-size 4 --limit 16 --output runs/csg-elife
```

These short runs check execution, not benchmark accuracy. Remove `--limit`
and the reduced epoch/width overrides for full-data training. Each run saves
its configuration, metrics, and validation-selected checkpoint under `--output`.
Use a separate output directory for each run.

## Verification and scope

```bash
python verify.py
python verify_conditions.py
```

This release provides software examples on SHD and CIFAR-10. It does not by
itself reproduce all manuscript results or constitute a device-level on-chip
learning demonstration. Training and preprocessing details, source conventions,
and implementation notes are provided in [SOURCE_NOTES.md](SOURCE_NOTES.md).

## Reference

Sungmin Lee, Jae-Hyun Lee, and colleagues. *Hardware-Aware On-Chip Learning
with a Scalable Single-Transistor-based Neuromorphic Platform*. Manuscript.
