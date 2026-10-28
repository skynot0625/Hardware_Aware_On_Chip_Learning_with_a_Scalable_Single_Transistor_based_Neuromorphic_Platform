"""SOLO + cSG entry point corresponding to the original model_current.py."""

from .resnet import PreActResNet18


class SOLO_PreactResNet18(PreActResNet18):
    def __init__(self, scale: int = 64, num_steps: int = 5) -> None:
        super().__init__(scale=scale, num_steps=num_steps, elife=False)
