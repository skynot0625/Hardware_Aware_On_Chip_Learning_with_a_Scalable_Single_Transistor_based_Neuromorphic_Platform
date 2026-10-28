"""Re-export original class text, without AST rewriting or architecture refactoring."""

import argparse
import ast
import hashlib
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-dir", type=Path, required=True)
    parser.add_argument("--analog-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    header = ('# Source-preserving export; see SOURCE_PROVENANCE.json and export_originals.py.\n'
              '# noqa: SIZE_OK - original research classes retained verbatim for review.\n'
              'import random\nimport torch\nimport torch.nn as nn\n'
              'import torch.nn.init as init\nfrom . import surrogate\n\n\n')
    helpers = ["LIF_Node", "cTRACE_Node", "LI_Node"]
    for filename in ["model.py", "model_current.py", "model_current_eLIFE.py"]:
        names = helpers + (["eLIF_Node_SOLOplus"] if "eLIFE" in filename else [])
        body = portable(extract(args.solo_dir / filename, names + ["SOLO_PreactResNet18"]))
        if "eLIFE" in filename:
            body = body.replace('        LIF_U = I_in\n', '')
        (args.output / filename).write_text(header + body)
        if "eLIFE" in filename:
            sg = body.replace('surrogate.HeavisideCurrentBoxcarCall', 'surrogate.HeavisideBoxcarCall')
            sg = sg.replace('self.surrogate_function(LIF_U, I_in)', 'self.surrogate_function(LIF_U)')
            (args.output / "model_eLIFE.py").write_text(header + sg)
    shd_helpers = ["LIF_Node_SOLOplus", "LIF_current_Node_SOLOplus", "htLIF_Node_SOLOplus",
                   "htLIF_current_Node_SOLOplus", "cTRACE_Node_SOLOplus", "eLIF_Node_SOLOplus"]
    classes = ["mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3",
               "mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3",
               "mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_pure",
               "mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_pure"]
    body = portable(extract(args.analog_model, shd_helpers + classes))
    body = body.replace('AnalogLinear(', 'nn.Linear(').replace(', rpu_config=rpu_config', '')
    (args.output / "model_shd.py").write_text(header + body)
    surrogate = (args.solo_dir / "surrogate.py").read_text()
    surrogate = surrogate.replace('        return grad_input, None, None\n\n\nclass Heaviside_eLIFCall',
                                  '        return grad_input, None\n\n\nclass Heaviside_eLIFCall')
    (args.output / "surrogate.py").write_text(surrogate)
    sources = [args.solo_dir / name for name in ["model.py", "model_current.py", "model_current_eLIFE.py", "surrogate.py"]]
    sources.append(args.analog_model)
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (args.output.parent / "SOURCE_PROVENANCE.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
