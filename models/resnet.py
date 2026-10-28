"""Compact SOLO pre-activation ResNet-18, extracted from the two current models."""

import math

import torch
from torch import Tensor, nn

from .neurons import Eligibility, LIFState, Surrogate, lif_step, project


class ResidualBlock(nn.Module):
    """Two pre-activation spiking convolutions and the original skip branch."""

    def __init__(self, channels: tuple[int, int], stride: int, projection: bool) -> None:
        super().__init__()
        before, after = channels
        self.bn1 = nn.BatchNorm2d(before)
        self.conv1 = nn.Conv2d(before, after, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(after)
        self.conv2 = nn.Conv2d(after, after, 3, 1, 1, bias=False)
        self.skip = (
            nn.Conv2d(before, after, 1, stride, bias=False)
            if projection else nn.Sequential(nn.BatchNorm2d(before))
        )


class PreActResNet18(nn.Module):
    """SOLO cSG baseline or cSG+eLIFE; inputs [B,3,32,32], outputs [B,10].

    Defaults follow configs/model/SOLO_current_eLIFE.py. Earlier steps have no
    graph; only the last step contributes spatial gradients. Logits are summed
    over time, as in the source. States are local to each forward call.
    """

    def __init__(self, scale: int = 64, num_steps: int = 5, elife: bool = False,
                 surrogate: Surrogate = Surrogate.CSG) -> None:
        super().__init__()
        if scale < 1 or num_steps < 1:
            raise ValueError("scale and num_steps must be positive")
        self.scale = scale
        self.num_steps = num_steps
        self.elife = elife
        self.surrogate = Surrogate(surrogate)
        self.register_buffer("decay", torch.tensor(1 / (1 + math.exp(-1))))
        self.conv1 = nn.Conv2d(3, scale, 3, padding=1, bias=False)
        widths = [scale, scale, scale * 2, scale * 2, scale * 4, scale * 4,
                  scale * 8, scale * 8]
        before = scale
        blocks = []
        for index, after in enumerate(widths):
            stride = 2 if after != before else 1
            blocks.append(ResidualBlock((before, after), stride, index % 2 == 0))
            before = after
        self.blocks = nn.ModuleList(blocks)
        self.bn1 = nn.BatchNorm2d(scale * 8)
        self.pool1 = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(scale * 8, 100, bias=False)
        self.fc2 = nn.Linear(100, 100, bias=False)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("expected images with shape [batch, 3, 32, 32]")
        states: dict[int, LIFState] = {}
        traces: dict[int, Eligibility] = {}
        input_trace = torch.zeros_like(images)
        total = images.new_zeros((images.shape[0], 10))
        outer_grad = torch.is_grad_enabled()

        def fire(current: Tensor, index: int) -> Tensor:
            if index not in states:
                # Preserve the source's chained initialization: both start at .5.
                initial = torch.full_like(current, 0.5)
                states[index] = LIFState(initial, initial)
            states[index] = lif_step(current, states[index], self.decay, self.surrogate)
            return states[index].spike

        def encode(spike: Tensor, index: int) -> Tensor | None:
            if not self.elife:
                return None
            previous = traces.get(index)
            if previous is None:
                previous = Eligibility.zeros(spike)
            traces[index] = previous.advance(spike)
            return traces[index].encode(spike)

        for step in range(self.num_steps):
            with torch.set_grad_enabled(outer_grad and step == self.num_steps - 1):
                if self.elife:
                    input_trace = (self.decay * input_trace.detach() + images).clamp_max(100)
                    value = project(self.conv1, images, input_trace)
                else:
                    value = self.conv1(images)
                for index, block in enumerate(self.blocks):
                    first = fire(block.bn1(value), index * 2)
                    current = project(block.conv1, first, encode(first, index * 2))
                    second = fire(block.bn2(current), index * 2 + 1)
                    current = project(block.conv2, second, encode(second, index * 2 + 1))
                    value = current + block.skip(value)
                spike = fire(self.bn1(value), 16)
                eligible = encode(spike, 16)
                pooled = self.pool1(spike).flatten(1)
                if eligible is not None:
                    current = project(self.fc1, pooled, self.pool1(eligible).flatten(1))
                else:
                    current = self.fc1(pooled)
                output_spike = fire(current, 17)
                total = total + self.fc2(output_spike).reshape(-1, 10, 10).mean(2)
        return total
