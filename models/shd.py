"""Digital SHD feedback network adapted from model_analog.py's cSG+eLIFE class."""

import random

import torch
from torch import Tensor, nn

from .neurons import Eligibility, LIFState, Surrogate, lif_step as update_lif, project as synapse


class SHDELife(nn.Module):
    """Five spiking stages, two recurrent paths, residuals and four feedback paths.

    Input: [B,T,700] or [B,T,1,700]. Output: [B,20]. Default T=20, width=1024.
    All synapses are digital and bias-free. SOLO trains on a random prefix of
    at least four frames; evaluation uses every frame. Only its final step has
    gradients. Unlike the source loop, no penultimate frame is skipped.
    """

    def __init__(self, scale: int = 1024, num_steps: int = 20, min_steps: int = 4,
                 surrogate: Surrogate = Surrogate.CSG, elife: bool = True) -> None:
        super().__init__()
        if scale < 1 or not 1 <= min_steps <= num_steps:
            raise ValueError("require scale > 0 and 1 <= min_steps <= num_steps")
        self.scale = scale
        self.num_steps = num_steps
        self.min_steps = min_steps
        self.surrogate = Surrogate(surrogate)
        self.elife = elife
        self.fc = nn.Linear(700, scale, bias=False)
        self.fc1 = nn.Linear(scale, scale, bias=False)
        self.fc2 = nn.Linear(scale, scale, bias=False)
        self.fc3 = nn.Linear(scale, scale, bias=False)
        self.rec = nn.Linear(scale, scale, bias=False)
        self.rec2rec = nn.Linear(scale, scale, bias=False)
        self.rec2fc = nn.Linear(scale, scale, bias=False)
        self.rec_skip = nn.Linear(scale, scale, bias=False)
        self.skip = nn.Linear(scale, scale, bias=False)
        self.fb0 = nn.Linear(scale, scale, bias=False)
        self.fb1 = nn.Linear(scale, scale, bias=False)
        self.fb2 = nn.Linear(scale, scale, bias=False)
        self.fb3 = nn.Linear(scale, scale, bias=False)
        self.outw = nn.Linear(scale, 200, bias=False)
        for layer in (self.fc1, self.fc2, self.fc3, self.rec, self.rec2rec,
                      self.rec2fc, self.rec_skip, self.skip, self.fb0,
                      self.fb1, self.fb2, self.fb3, self.outw):
            nn.init.xavier_uniform_(layer.weight)
        # Preserve effective initialization: original helpers overwrite logits to 0.
        # Stages 1 and 3 have per-neuron trainable decay; the others are scalar.
        self.decay_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(scale if i in (1, 3) else 1)) for i in range(5)
        ])

    def forward(self, inputs: Tensor) -> Tensor:
        frames = inputs.squeeze(2) if inputs.ndim == 4 else inputs
        if frames.ndim != 3 or tuple(frames.shape[1:]) != (self.num_steps, 700):
            raise ValueError(f"expected [B,{self.num_steps},700] or [B,{self.num_steps},1,700]")
        initial = frames.new_full((frames.shape[0], self.scale), 0.5)
        states = [LIFState(initial, initial) for _ in range(5)]
        traces = [Eligibility.zeros(initial) for _ in range(5)]
        input_trace = frames.new_zeros((frames.shape[0], 700))
        total = frames.new_zeros((frames.shape[0], 20))
        steps = random.randint(self.min_steps, self.num_steps) if self.training else self.num_steps
        outer_grad = torch.is_grad_enabled()

        def project(layer: nn.Module, spike: Tensor, eligible: Tensor) -> Tensor:
            return synapse(layer, spike, eligible if self.elife else None)

        def lif_step(current: Tensor, state: LIFState, decay: Tensor) -> LIFState:
            return update_lif(current, state, decay, self.surrogate)

        for step in range(steps):
            with torch.set_grad_enabled(outer_grad and step == steps - 1):
                previous = [state.spike.detach() for state in states]
                old_eligible = [trace.encode(spike) for trace, spike in zip(traces, previous)]
                frame = frames[:, step]
                input_trace = 0.78 * input_trace.detach() + frame
                current = project(self.fc, frame, input_trace)
                current = current + project(self.fb0, previous[4], old_eligible[4])
                states[0] = lif_step(current, states[0], self.decay_logits[0].sigmoid())
                traces[0] = traces[0].advance(states[0].spike)
                s0 = states[0].spike
                e0 = traces[0].encode(s0)

                current = project(self.fc1, s0, e0)
                current = current + project(self.fb1, previous[4], old_eligible[4])
                current = current + project(self.rec, previous[1], old_eligible[1])
                states[1] = lif_step(current, states[1], self.decay_logits[1].sigmoid())
                traces[1] = traces[1].advance(states[1].spike)
                s1 = states[1].spike

                current = project(self.fc2, s1, traces[1].encode(s1))
                current = current + project(self.fb2, previous[4], old_eligible[4])
                current = current + project(self.skip, s0, e0)
                states[2] = lif_step(current, states[2], self.decay_logits[2].sigmoid())
                traces[2] = traces[2].advance(states[2].spike)
                s2 = states[2].spike
                e2 = traces[2].encode(s2)

                current = project(self.fc3, s2, e2)
                current = current + project(self.rec2rec, previous[3], old_eligible[3])
                current = current + project(self.fb3, previous[4], old_eligible[4])
                states[3] = lif_step(current, states[3], self.decay_logits[3].sigmoid())
                traces[3] = traces[3].advance(states[3].spike)
                s3 = states[3].spike

                current = project(self.rec2fc, s3, traces[3].encode(s3))
                current = current + project(self.rec_skip, s2, e2)
                states[4] = lif_step(current, states[4], self.decay_logits[4].sigmoid())
                traces[4] = traces[4].advance(states[4].spike)
                total = total + self.outw(states[4].spike).reshape(-1, 20, 10).mean(2)
        return total / steps
