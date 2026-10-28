# Shared neuron helpers for the SHD-specialized Arch1/Arch2 examples.
# eP = eligible potential; its input trace is retained for learning.
# noqa: SIZE_OK - retain original explicit research forward methods.
import torch
import torch.nn as nn
from . import surrogate


class LIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        self.LIF_decay = LIF_decay
        LIF_decay = torch.logit(LIF_decay)
        LIF_decay = torch.tensor(0, dtype=torch.float32)
        self.LIF_decay = nn.Parameter(LIF_decay)
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class LIF_cSG_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideCurrentBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        LIF_decay = torch.logit(LIF_decay)
        LIF_decay = torch.tensor(0, dtype=torch.float32)
        self.LIF_decay = LIF_decay
        self.LIF_decay = nn.Parameter(LIF_decay)
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U, I_in)
        return LIF_U, LIF_S

class htLIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, scale, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()

        median = 4.0  # Target median in ms
        mu = torch.log(torch.tensor(median, dtype=torch.float32))  # [1][3][7]
        sigma = 0.5  # Distribution width

        lognormal_dist = torch.distributions.LogNormal(loc=mu, scale=sigma)  # [1][2][3]
        samples = lognormal_dist.sample((scale,))  # Sample shape (512,) [1][2][6]

        decay_values = torch.exp(-1 / samples)  # [1][3][5]
        decay_values = torch.logit(decay_values)
        decay_values = torch.zeros_like(samples)
        self.LIF_decay = decay_values
        self.LIF_decay = nn.Parameter(decay_values)
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class htLIF_cSG_Node_SOLOplus(nn.Module):
    def __init__(
        self, scale, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideCurrentBoxcarCall()
    ):
        super().__init__()

        median = 4.0  # Target median in ms
        mu = torch.log(torch.tensor(median, dtype=torch.float32))  # [1][3][7]
        sigma = 0.5  # Distribution width
        sigma = 1.0  # Distribution width

        lognormal_dist = torch.distributions.LogNormal(loc=mu, scale=sigma)  # [1][2][3]
        samples = lognormal_dist.sample((scale,))  # Sample shape (512,) [1][2][6]

        decay_values = torch.exp(-1 / samples)  # [1][3][5]
        decay_values = torch.ones_like(samples) * 0.78
        decay_values = torch.logit(decay_values)
        decay_values = torch.zeros_like(samples)
        self.LIF_decay = decay_values
        self.LIF_decay = nn.Parameter(decay_values)
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U, I_in)
        return LIF_U, LIF_S

class eP_Node_SOLOplus(nn.Module):
    def __init__(
        self,
        eP_decay: torch.Tensor,
        eP_a_in: torch.Tensor,
        eP_clamp_ceiling: torch.Tensor,
    ):
        super().__init__()
        self.eP_decay = eP_decay
        self.eP_a_in = eP_a_in
        self.eP_clamp_ceiling = eP_clamp_ceiling

    def forward(self, eP_U: torch.Tensor, S_in: torch.Tensor):
        eP_U = self.eP_decay * eP_U + self.eP_a_in * S_in
        return eP_U

class eLIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        self.LIF_decay = LIF_decay
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S
