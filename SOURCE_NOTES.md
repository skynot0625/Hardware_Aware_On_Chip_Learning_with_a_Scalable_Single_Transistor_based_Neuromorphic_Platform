# Source and implementation notes

## Source preservation

Model classes are extracted as source text from `src/SOLO/model.py`,
`model_current.py`, `model_current_eLIFE.py`, and the supplied `model_analog.py`.
The final definition of duplicate class names is used, matching Python import
semantics. Surviving layer names, equations, and explicit forward order remain
unchanged. Original filenames and SHA-256 hashes are in `SOURCE_PROVENANCE.json`.

```bash
python export_originals.py --solo-dir /path/to/ml/src/SOLO --analog-model /path/to/model_analog.py
```

`export_originals.py` records the precise cleanup rules. It removes enumerated
dead statements using their source-line spans; it does not unparse or rebuild
the surviving model computation. Shared SHD neuron helpers are in `shd_nodes.py`.

## Minimal adaptations

- Replace template imports with local imports and omit unused dependencies.
- Let the caller choose the device instead of moving modules in constructors.
- Replace SHD `AnalogLinear` with `nn.Linear`, removing `rpu_config`.
- Remove the ResNet clone's `LIF_U = I_in` overwrite so leak/integration remains.
- Correct `eLIF.backward` to return two gradients for two inputs.
- Derive ResNet SG+eLIFE from cSG+eLIFE by changing the main-neuron surrogate
  from current-domain to voltage-domain. It is explicitly a derived variant.

## Dead-code cleanup

- ResNet SG/cSG: remove the unused `cTRACE_Node` and `LI_Node` classes,
  unused LI instance, and unused initial convolution-state allocation.
- ResNet eLIFE: retain the input trace as `eP_Node` (eligible potential); remove unused LI and
  the unused first pair of clone instances.
- SHD: remove intermediate `cTRACE_1`–`cTRACE_4` and their trace states, unused
  `fb`/`LIF_fb`/`LIF_rec` helpers, unused clone states, and unused thresholds.
  Pure variants have no input trace or clone modules. Arch1 has no feedback
  projections; its unused output clone is also removed.
- Remove commented-out experiments and unused surrogate classes.

Removing unused randomly initialized layers changes RNG consumption during
construction. Thus same-seed initial weights need not equal old releases;
behavioral equivalence was tested by loading the same retained weights.
Old state dicts can contain removed module keys: use only matching model keys
when transferring checkpoints and investigate any missing active weights.

## Architectures and datasets

`model_Arch1.py` uses the original `_x` variants with recurrent/residual
connections and no feedback. `model_Arch2.py` uses the original four-feedback
variants. Public names are `Arch1_SG`, `Arch1_cSG`, `Arch1_SG_eLIFE`,
`Arch1_cSG_eLIFE` and the corresponding `Arch2_*` names. In the original
source, `_pure` denotes no eLIFE and `_current_` denotes cSG; `fb3_x` maps
to Arch1 and `fb3` maps to Arch2. Despite their historical `smnist` names,
these supplied classes are SHD-specialized (700 input channels, 20 steps, 20 classes).
Neither filename implies that an F-MNIST or S-MNIST input adapter is included.

Input trace identifiers are renamed from `cTRACE`/`eU` to `eP` (eligible
potential). `eP_Node` and `eP_Node_SOLOplus` retain the original trace equations;
this is a naming change, not removal of the input-gradient path. ResNet filenames
uniformly use `model_ResNet18_*`.

## Retained source conventions

- The SHD loop uses `range(init_step, last_step - 1)` and skips the penultimate
  frame. This source behavior is documented rather than silently repaired.
- Aliased initial main-neuron membrane/spike tensors both start at 0.5.
- SHD main-neuron decay logits initialize to zero in the original helpers;
  the fixed eLIFE clone coefficients are separate and are not sigmoid logits.
- Original variants have different initialization routines. These examples
  alone are not a newly controlled accuracy ablation.
- Firing uses strict `>`, and boxcar windows use strict `<` at their boundaries.
- The eLIFE-enabled digital input projection uses a continuous input trace.
  Removing it would change weight gradients, not merely simplify dead code.

## Runner differences

The standalone CIFAR loader preserves official test separation instead of the
original template's pooled split. Adam/cosine settings follow Supplementary S4.1;
the generic Hydra experiment's `lr=0.01` override is not used. No Cutout is added.
SHD event binning is a demonstration choice, not a recovered original pipeline.
Full paper accuracy, statistical comparisons and analog behavior are not
validated by these smoke examples.
