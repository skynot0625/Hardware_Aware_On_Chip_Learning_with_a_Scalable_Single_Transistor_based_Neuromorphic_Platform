"""Current surrogate gradients and fixed eLIFE dynamics (Methods, Eqs. 3–11)."""

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class Surrogate(str, Enum):
    SG = "sg"
    CSG = "csg"


def current_spike(voltage: Tensor, current: Tensor, half_width: float = 1.0) -> Tensor:
    """Hard forward spike; current-domain boxcar gradient through voltage only.

    The source convention is strict > at firing and strict < at window edges.
    The detached mask avoids a second gradient path through synaptic current.
    """
    hard = (voltage > 1.0).to(voltage.dtype)
    mask = ((current.detach() - 1.0).abs() < half_width).to(voltage.dtype)
    return hard + (voltage - voltage.detach()) * mask


def learning_path(forward_value: Tensor, backward_value: Tensor) -> Tensor:
    """Keep physical forward values but use the eligibility path for gradients."""
    return forward_value.detach() + (backward_value - backward_value.detach())


@dataclass(frozen=True, slots=True)
class LIFState:
    voltage: Tensor
    spike: Tensor


def lif_step(current: Tensor, state: LIFState, decay: Tensor,
             surrogate: Surrogate = Surrogate.CSG) -> LIFState:
    """Spatial-only LIF update with detached previous-time states."""
    voltage = decay * state.voltage.detach() * (1 - state.spike.detach()) + current
    signal = voltage if surrogate is Surrogate.SG else current
    return LIFState(voltage, current_spike(voltage, signal))


@dataclass(frozen=True, slots=True)
class Eligibility:
    voltage1: Tensor
    spike1: Tensor
    voltage2: Tensor
    spike2: Tensor

    @classmethod
    def zeros(cls, reference: Tensor) -> "Eligibility":
        zero = torch.zeros_like(reference)
        return cls(zero, zero, zero, zero)

    def advance(self, main: Tensor) -> "Eligibility":
        """Simultaneous delayed coupling; beta1=.5, beta2=1, unit weights.

        Main-text Eqs. (7)–(10), Supplementary S4.2. Clone states are local
        and never part of the spatial gradient graph.
        """
        v1 = 0.5 * self.voltage1 * (1 - self.spike1) + self.spike2 + main.detach()
        v2 = self.voltage2 * (1 - self.spike2) + self.spike1
        return Eligibility(v1, (v1 > 1).to(main.dtype), v2, (v2 > 1).to(main.dtype))

    def encode(self, main: Tensor) -> Tensor:
        """Eq. (11): OR in forward, identity gradient to main spikes only."""
        eligible = torch.maximum(torch.maximum(self.spike1, self.spike2), main.detach())
        return main + (eligible - main).detach()


def project(layer: nn.Module, spike: Tensor, eligible: Tensor | None) -> Tensor:
    """Evaluate a synapse with optional backward-only presynaptic eligibility."""
    if eligible is None or not torch.is_grad_enabled():
        return layer(spike)
    return learning_path(layer(spike), layer(eligible))
