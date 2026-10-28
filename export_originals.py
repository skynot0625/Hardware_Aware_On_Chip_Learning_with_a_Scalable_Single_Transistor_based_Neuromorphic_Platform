"""Export original class text with explicit dead-code removal and dataset headers."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def extract(path: Path, names: list[str]) -> str:
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    classes = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.ClassDef)}
    return "\n\n".join("".join(lines[classes[name].lineno - 1:classes[name].end_lineno]).rstrip()
                       for name in names) + "\n"


def portable(text: str) -> str:
    text = text.replace('        self.to("cuda" if torch.cuda.is_available() else "cpu")\n', '')
    return text.replace('.to("cuda" if torch.cuda.is_available() else "cpu")', '')


def public_names(text: str) -> str:
    renamed = text.replace("cTRACE", "eP").replace("TRACE", "eP").replace("LIF_current", "LIF_cSG")
    return re.sub(r"\beU\b", "eP", renamed)


def prune(text: str, attributes: set[str], locals_: set[str]) -> str:
    """Remove only enumerated dead statements; keep surviving source lines intact."""
    lines = text.splitlines()
    removed: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Assign, ast.Expr)):
            continue
        dead = any(
            (isinstance(part, ast.Attribute) and isinstance(part.value, ast.Name)
             and part.value.id == "self" and part.attr in attributes)
            or (isinstance(part, ast.Name) and part.id in locals_)
            for part in ast.walk(node)
        )
        if dead:
            removed.update(range(node.lineno - 1, node.end_lineno))
    result = [line.rstrip() for index, line in enumerate(lines)
              if index not in removed and not line.lstrip().startswith("#")]
    return "\n".join(result).strip() + "\n"


def clean_shd(text: str, elife: bool, feedback: bool) -> str:
    attributes = {"fb", "LIF_fb", "LIF_rec", "eLIF_fb_1", "eLIF_fb_2",
                  "time_resolution", "ethreshold", "ethreshold1", "ethreshold2",
                  "ethreshold3", "ethreshold4"}
    attributes.update(f"cTRACE_{i}" for i in range(1, 5))
    locals_ = {"fb_U", "fb_S"}
    locals_.update(f"h{i}_e{kind}" for i in range(1, 5) for kind in ("U", "S"))
    locals_.update(f"eLIF_fb_{suffix}" for suffix in ("U1", "U2", "S1", "S2", "S1_prev", "S2_prev"))
    if not feedback:
        attributes.update(f"fb{i}" for i in range(4))
    if not elife:
        attributes.add("cTRACE")
        locals_.update({"eU", "eS"})
    for i in range(1, 6):
        if not elife or (i == 5 and not feedback):
            attributes.update({f"eLIF_{i}_1", f"eLIF_{i}_2"})
            locals_.update(f"eLIF_h{i}_{suffix}" for suffix in ("U1", "U2", "S1", "S2", "S1_prev", "S2_prev"))
    return prune(text, attributes, locals_)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-dir", type=Path, required=True)
    parser.add_argument("--analog-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    header = ('# noqa: SIZE_OK - retain original explicit research forward methods.\n'
              'import torch\nimport torch.nn as nn\nfrom . import surrogate\n\n\n')
    resnet_data = ('# Dataset: CIFAR-10 in this release; input [B,3,32,32], output [B,10].\n'
                   '# Paper ResNet family: CIFAR-10/100, CIFAR10DVS, DVS128Gesture.\n'
                   '# Event-dataset frontends and benchmark runners are not included here.\n')
    resnet_names = {"model.py": "model_ResNet18_SG.py",
                    "model_current.py": "model_ResNet18_cSG.py",
                    "model_current_eLIFE.py": "model_ResNet18_cSG_eLIFE.py"}
    for filename, destination in resnet_names.items():
        names = ["LIF_Node"] + (["cTRACE_Node", "eLIF_Node_SOLOplus"] if "eLIFE" in filename else [])
        body = portable(extract(args.solo_dir / filename, names + ["SOLO_PreactResNet18"]))
        if "eLIFE" in filename:
            body = body.replace('        LIF_U = I_in\n', '')
        body = prune(body, {"LI", "eLIF_1_1", "eLIF_1_2"}, {"c1_mem", "c1_spike"})
        body = public_names(body)
        if "eLIFE" in filename:
            body = '# eP = eligible potential: input trace used for first-layer weight gradients.\n' + body
        (args.output / destination).write_text(resnet_data + header + body)
        if "eLIFE" in filename:
            sg = body.replace('surrogate.HeavisideCurrentBoxcarCall', 'surrogate.HeavisideBoxcarCall')
            sg = sg.replace('self.surrogate_function(LIF_U, I_in)', 'self.surrogate_function(LIF_U)')
            (args.output / "model_ResNet18_SG_eLIFE.py").write_text(resnet_data + header + sg)
    shd_helpers = ["LIF_Node_SOLOplus", "LIF_current_Node_SOLOplus", "htLIF_Node_SOLOplus",
                   "htLIF_current_Node_SOLOplus", "cTRACE_Node_SOLOplus", "eLIF_Node_SOLOplus"]
    nodes = prune(portable(extract(args.analog_model, shd_helpers)), set(), set())
    (args.output / "shd_nodes.py").write_text('# Shared neuron helpers for the SHD-specialized Arch1/Arch2 examples.\n# eP = eligible potential; its input trace is retained for learning.\n' + header + public_names(nodes))
    for architecture in (1, 2):
        body = ''
        for sg in (False, True):
            for elife in (False, True):
                name = ('mlp4_SOLOplus_' + ('' if sg else 'current_')
                        + 'smnist_RES_RSNN_ht_res_fb3' + ('_x' if architecture == 1 else '')
                        + ('' if elife else '_pure'))
                source = portable(extract(args.analog_model, [name]))
                source = source.replace('AnalogLinear(', 'nn.Linear(').replace(', rpu_config=rpu_config', '')
                source = source.replace('        rpu_config = None,\n', '')
                public_class = f'Arch{architecture}_' + ('SG' if sg else 'cSG') + ('_eLIFE' if elife else '')
                source = clean_shd(source, elife, architecture == 2).replace(f'class {name}(', f'class {public_class}(')
                body += public_names(source) + '\n\n'
        data_header = (f'# Architecture {architecture}: recurrent/residual network '
                       + ('without feedback.\n' if architecture == 1 else 'with four output-feedback projections.\n')
                       + '# Dataset: SHD; input [B,20,700] or [B,20,1,700], output [B,20].\n'
                       + '# Paper small-network family: F-MNIST, S-MNIST, SHD. This file is SHD-specialized.\n')
        data_header += '# eP = eligible potential: input trace present only in eLIFE-enabled variants.\n'
        imports = ('import random\nimport torch.nn.init as init\nfrom .shd_nodes import (\n'
                   + ''.join(f'    {public_names(name)},\n' for name in shd_helpers) + ')\n\n')
        (args.output / f"model_Arch{architecture}.py").write_text(data_header + header + imports + body.rstrip() + '\n')
    surrogate_names = ["Boxcar", "HeavisideBoxcarCall", "cTRACEClamp", "cTraceSwitch",
                       "CurrentBoxcar", "HeavisideCurrentBoxcarCall", "eLIF",
                       "Heaviside_eLIFCall", "eLIF_fc"]
    surrogate = extract(args.solo_dir / "surrogate.py", surrogate_names)
    surrogate = surrogate.replace('        return grad_input, None, None\n\n\nclass Heaviside_eLIFCall',
                                  '        return grad_input, None\n\n\nclass Heaviside_eLIFCall')
    surrogate = prune(surrogate, set(), set())
    (args.output / "surrogate.py").write_text('import torch\nimport torch.nn as nn\n\n' + public_names(surrogate))
    sources = [args.solo_dir / name for name in ["model.py", "model_current.py", "model_current_eLIFE.py", "surrogate.py"]]
    sources.append(args.analog_model)
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (args.output.parent / "SOURCE_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
