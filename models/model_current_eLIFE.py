"""SOLO + cSG + eLIFE, using the manuscript's fixed clone dynamics."""

from .resnet import PreActResNet18


class SOLO_PreactResNet18(PreActResNet18):
    def __init__(self, scale: int = 64, num_steps: int = 5) -> None:
        super().__init__(scale=scale, num_steps=num_steps, elife=True)
