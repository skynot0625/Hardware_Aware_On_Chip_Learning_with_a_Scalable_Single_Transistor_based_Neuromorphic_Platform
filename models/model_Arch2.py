# Architecture 2: recurrent/residual network with four output-feedback projections.
# Dataset: SHD; input [B,20,700] or [B,20,1,700], output [B,20].
# Paper small-network family: F-MNIST, S-MNIST, SHD. This file is SHD-specialized.
# eP = eligible potential: input trace present only in eLIFE-enabled variants.
# noqa: SIZE_OK - retain original explicit research forward methods.
import torch
import torch.nn as nn
from . import surrogate


import random
import torch.nn.init as init
from .shd_nodes import (
    LIF_Node_SOLOplus,
    LIF_cSG_Node_SOLOplus,
    htLIF_Node_SOLOplus,
    htLIF_cSG_Node_SOLOplus,
    eP_Node_SOLOplus,
    eLIF_Node_SOLOplus,
)

class Arch2_cSG(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_eP_decay: float = 0.78,
        init_eP_a: float = 1,
        init_eP_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps


        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc1 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF1 = htLIF_cSG_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc2 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF2 = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc3 = nn.Linear(self.scale, 1024, bias=False)
        self.LIF3 = htLIF_cSG_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )


        self.rec2rec = nn.Linear(1024, 1024, bias=False)

        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.rec_skip = nn.Linear(self.scale, 1024, bias=False)


        self.fb0 = nn.Linear(1024, self.scale, bias=False)
        self.fb1 = nn.Linear(1024, self.scale, bias=False)
        self.fb2 = nn.Linear(1024, self.scale, bias=False)
        self.fb3 = nn.Linear(1024, 1024, bias=False)

        self.skip = nn.Linear(self.scale, self.scale, bias=False)

        self.rec = nn.Linear(self.scale, self.scale, bias=False)


        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.rec2rec.weight)
        init.xavier_uniform_(self.rec2fc.weight)
        init.xavier_uniform_(self.rec_skip.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)

        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)

        boost1 = torch.zeros(batch_size, 20, device=self.device)
        acc_mem = torch.zeros(batch_size, 20, device=self.device)



        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():

                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))

                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1

        I_in = self.fc(x[last_step])
        I_in_fb0 = self.fb0(out_S)
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        I_in = self.fc1(S)
        I_in_fb1 = self.fb1(out_S)
        I_in_rec = self.rec(h1_S)
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        I_in = self.fc2(h1_S)
        I_in_fb2 = self.fb2(out_S)
        I_in_skip = self.skip(S)
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        I_in = self.fc3(h2_S)
        I_in_rec = self.rec2rec(h3_S)
        I_in_fb3 = self.fb3(out_S)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        I_in = self.rec2fc(h3_S)
        I_in_skip = self.rec_skip(h2_S)
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps


class Arch2_cSG_eLIFE(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_eP_decay: float = 0.78,
        init_eP_a: float = 1,
        init_eP_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = 1024
        self.subthresh = subthresh
        self.min_steps = min_steps


        self.eP = eP_Node_SOLOplus(
            eP_decay=init_eP_decay,
            eP_a_in=init_eP_a,
            eP_clamp_ceiling=init_eP_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_1_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_1_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc1 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF1 = htLIF_cSG_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_2_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_2_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc2 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF2 = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_3_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_3_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc3 = nn.Linear(self.scale, 1024, bias=False)
        self.LIF3 = htLIF_cSG_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_4_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_4_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )

        self.rec2rec = nn.Linear(1024, 1024, bias=False)

        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_cSG_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_5_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_5_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )

        self.rec_skip = nn.Linear(self.scale, 1024, bias=False)


        self.fb0 = nn.Linear(1024, self.scale, bias=False)
        self.fb1 = nn.Linear(1024, self.scale, bias=False)
        self.fb2 = nn.Linear(1024, self.scale, bias=False)
        self.fb3 = nn.Linear(1024, 1024, bias=False)

        self.skip = nn.Linear(self.scale, self.scale, bias=False)

        self.rec = nn.Linear(self.scale, self.scale, bias=False)


        self.outw = nn.Linear(self.scale, 200, bias=False)



        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eP = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)

        boost1 = torch.zeros(batch_size, 20, device=self.device)
        acc_mem = torch.zeros(batch_size, 20, device=self.device)

        eLIF_h1_U1 = eLIF_h1_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h1_U2 = eLIF_h1_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_U1 = eLIF_h2_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_U2 = eLIF_h2_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_U1 = eLIF_h3_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_U2 = eLIF_h3_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_U1 = eLIF_h4_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_U2 = eLIF_h4_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_U1 = eLIF_h5_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_U2 = eLIF_h5_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]


            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():

                eP = self.eP(eP.detach(), x[step])
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1, eLIF_h1_S1, S * 1 + eLIF_h1_S2_prev * 1)
                eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2, eLIF_h1_S2, eLIF_h1_S1_prev * 1)
                eLIF_h1_S1_prev = eLIF_h1_S1
                eLIF_h1_S2_prev = eLIF_h1_S2
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1, eLIF_h2_S1, h1_S * 1 + eLIF_h2_S2_prev * 1)
                eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2, eLIF_h2_S2, eLIF_h2_S1_prev * 1)
                eLIF_h2_S1_prev = eLIF_h2_S1
                eLIF_h2_S2_prev = eLIF_h2_S2

                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1, eLIF_h3_S1, h2_S * 1 + eLIF_h3_S2_prev * 1)
                eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2, eLIF_h3_S2, eLIF_h3_S1_prev * 1)
                eLIF_h3_S1_prev = eLIF_h3_S1
                eLIF_h3_S2_prev = eLIF_h3_S2

                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
                eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
                eLIF_h4_S1_prev = eLIF_h4_S1
                eLIF_h4_S2_prev = eLIF_h4_S2

                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))
                eLIF_h5_U1, eLIF_h5_S1 = self.eLIF_5_1(eLIF_h5_U1, eLIF_h5_S1, out_S * 1 + eLIF_h5_S2_prev * 1)
                eLIF_h5_U2, eLIF_h5_S2 = self.eLIF_5_2(eLIF_h5_U2, eLIF_h5_S2, eLIF_h5_S1_prev * 1)
                eLIF_h5_S1_prev = eLIF_h5_S1
                eLIF_h5_S2_prev = eLIF_h5_S2

                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1



        eP = self.eP(eP.detach(), x[last_step])
        I_in = surrogate.eP_switch.apply(self.fc(x[last_step]).detach(), self.fc(eP))
        I_in_fb0 = surrogate.eP_switch.apply(self.fb0(out_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1.detach(), eLIF_h1_S1.detach(), S * 1 + eLIF_h1_S2_prev.detach() * 1)
        eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2.detach(), eLIF_h1_S2.detach(), eLIF_h1_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc1(S).detach(), self.fc1(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        I_in_fb1 = surrogate.eP_switch.apply(self.fb1(out_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_rec = surrogate.eP_switch.apply(self.rec(h1_S).detach(), self.rec(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1.detach(), eLIF_h2_S1.detach(), h1_S * 1 + eLIF_h2_S2_prev.detach() * 1)
        eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2.detach(), eLIF_h2_S2.detach(), eLIF_h2_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc2(h1_S).detach(), self.fc2(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        I_in_fb2 = surrogate.eP_switch.apply(self.fb2(out_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_skip = surrogate.eP_switch.apply(self.skip(S).detach(), self.skip(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1.detach(), eLIF_h3_S1.detach(), h2_S * 1 + eLIF_h3_S2_prev.detach() * 1)
        eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2.detach(), eLIF_h3_S2.detach(), eLIF_h3_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc3(h2_S).detach(), self.fc3(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        I_in_rec = surrogate.eP_switch.apply(self.rec2rec(h3_S).detach(), self.rec2rec(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        I_in_fb3 = surrogate.eP_switch.apply(self.fb3(out_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
        eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
        I_in = surrogate.eP_switch.apply(self.rec2fc(h3_S).detach(), self.rec2fc(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        I_in_skip = surrogate.eP_switch.apply(self.rec_skip(h2_S).detach(), self.rec_skip(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        eLIF_h5_U1, eLIF_h5_S1 = self.eLIF_5_1(eLIF_h5_U1, eLIF_h5_S1, out_S * 1 + eLIF_h5_S2_prev * 1)
        eLIF_h5_U2, eLIF_h5_S2 = self.eLIF_5_2(eLIF_h5_U2, eLIF_h5_S2, eLIF_h5_S1_prev * 1)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps


class Arch2_SG(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_eP_decay: float = 0.78,
        init_eP_a: float = 1,
        init_eP_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps


        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc1 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF1 = htLIF_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc2 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF2 = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.fc3 = nn.Linear(self.scale, 1024, bias=False)
        self.LIF3 = htLIF_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )


        self.rec2rec = nn.Linear(1024, 1024, bias=False)

        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.rec_skip = nn.Linear(self.scale, 1024, bias=False)


        self.fb0 = nn.Linear(1024, self.scale, bias=False)
        self.fb1 = nn.Linear(1024, self.scale, bias=False)
        self.fb2 = nn.Linear(1024, self.scale, bias=False)
        self.fb3 = nn.Linear(1024, 1024, bias=False)

        self.skip = nn.Linear(self.scale, self.scale, bias=False)

        self.rec = nn.Linear(self.scale, self.scale, bias=False)


        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.rec2rec.weight)
        init.xavier_uniform_(self.rec2fc.weight)
        init.xavier_uniform_(self.rec_skip.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)

        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)

        boost1 = torch.zeros(batch_size, 20, device=self.device)
        acc_mem = torch.zeros(batch_size, 20, device=self.device)



        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():

                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))

                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1

        I_in = self.fc(x[last_step])
        I_in_fb0 = self.fb0(out_S)
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        I_in = self.fc1(S)
        I_in_fb1 = self.fb1(out_S)
        I_in_rec = self.rec(h1_S)
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        I_in = self.fc2(h1_S)
        I_in_fb2 = self.fb2(out_S)
        I_in_skip = self.skip(S)
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        I_in = self.fc3(h2_S)
        I_in_rec = self.rec2rec(h3_S)
        I_in_fb3 = self.fb3(out_S)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        I_in = self.rec2fc(h3_S)
        I_in_skip = self.rec_skip(h2_S)
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps


class Arch2_SG_eLIFE(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_eP_decay: float = 0.78,
        init_eP_a: float = 1,
        init_eP_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps


        self.eP = eP_Node_SOLOplus(
            eP_decay=init_eP_decay,
            eP_a_in=init_eP_a,
            eP_clamp_ceiling=init_eP_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_1_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_1_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc1 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF1 = htLIF_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_2_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_2_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc2 = nn.Linear(self.scale, self.scale, bias=False)
        self.LIF2 = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_3_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_3_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.fc3 = nn.Linear(self.scale, 1024, bias=False)
        self.LIF3 = htLIF_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.eLIF_4_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_4_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )

        self.rec2rec = nn.Linear(1024, 1024, bias=False)

        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_5_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_5_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )

        self.rec_skip = nn.Linear(self.scale, 1024, bias=False)


        self.fb0 = nn.Linear(1024, self.scale, bias=False)
        self.fb1 = nn.Linear(1024, self.scale, bias=False)
        self.fb2 = nn.Linear(1024, self.scale, bias=False)
        self.fb3 = nn.Linear(1024, 1024, bias=False)

        self.skip = nn.Linear(self.scale, self.scale, bias=False)

        self.rec = nn.Linear(self.scale, self.scale, bias=False)

        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.rec2rec.weight)
        init.xavier_uniform_(self.rec2fc.weight)
        init.xavier_uniform_(self.rec_skip.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)


        self.boost1 = nn.AvgPool1d(10, 10)

        self.spike_records = {
        'LIF': [],      # fc → LIF
        'LIF1': [],     # fc1 → LIF1
        'LIF2': [],     # fc2 → LIF2
        'LIF3': [],     # fc3 → LIF3
        'LIF_out': [],  # rec2fc → LIF_out
        'Acc_mem': []
        }

    def forward(self, x):
        self.device = x.device
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eP = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)

        boost1 = torch.zeros(batch_size, 20, device=self.device)
        acc_mem = torch.zeros(batch_size, 20, device=self.device)

        eLIF_h1_U1 = eLIF_h1_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h1_U2 = eLIF_h1_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_U1 = eLIF_h2_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_U2 = eLIF_h2_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_U1 = eLIF_h3_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_U2 = eLIF_h3_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_U1 = eLIF_h4_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_U2 = eLIF_h4_S2 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_U1 = eLIF_h5_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_U2 = eLIF_h5_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for k in self.spike_records:
            self.spike_records[k].clear()

        for step in range(init_step, last_step - 1):
            with torch.no_grad():

                eP = self.eP(eP.detach(), x[step])
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1, eLIF_h1_S1, S * 1 + eLIF_h1_S2_prev * 1)
                eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2, eLIF_h1_S2, eLIF_h1_S1_prev * 1)
                eLIF_h1_S1_prev = eLIF_h1_S1
                eLIF_h1_S2_prev = eLIF_h1_S2
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1, eLIF_h2_S1, h1_S * 1 + eLIF_h2_S2_prev * 1)
                eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2, eLIF_h2_S2, eLIF_h2_S1_prev * 1)
                eLIF_h2_S1_prev = eLIF_h2_S1
                eLIF_h2_S2_prev = eLIF_h2_S2

                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1, eLIF_h3_S1, h2_S * 1 + eLIF_h3_S2_prev * 1)
                eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2, eLIF_h3_S2, eLIF_h3_S1_prev * 1)
                eLIF_h3_S1_prev = eLIF_h3_S1
                eLIF_h3_S2_prev = eLIF_h3_S2

                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
                eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
                eLIF_h4_S1_prev = eLIF_h4_S1
                eLIF_h4_S2_prev = eLIF_h4_S2

                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))
                eLIF_h5_U1, eLIF_h5_S1 = self.eLIF_5_1(eLIF_h5_U1, eLIF_h5_S1, out_S * 1 + eLIF_h5_S2_prev * 1)
                eLIF_h5_U2, eLIF_h5_S2 = self.eLIF_5_2(eLIF_h5_U2, eLIF_h5_S2, eLIF_h5_S1_prev * 1)
                eLIF_h5_S1_prev = eLIF_h5_S1
                eLIF_h5_S2_prev = eLIF_h5_S2

                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1

                self.spike_records['LIF'].append(S.detach().cpu().clone())
                self.spike_records['LIF1'].append(h1_S.detach().cpu().clone())
                self.spike_records['LIF2'].append(h2_S.detach().cpu().clone())
                self.spike_records['LIF3'].append(h3_S.detach().cpu().clone())
                self.spike_records['LIF_out'].append(out_S.detach().cpu().clone())
                self.spike_records['Acc_mem'].append(acc_mem.detach().cpu().clone())



        eP = self.eP(eP.detach(), x[last_step])
        I_in = surrogate.eP_switch.apply(self.fc(x[last_step]).detach(), self.fc(eP))
        I_in_fb0 = surrogate.eP_switch.apply(self.fb0(out_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1.detach(), eLIF_h1_S1.detach(), S * 1 + eLIF_h1_S2_prev.detach() * 1)
        eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2.detach(), eLIF_h1_S2.detach(), eLIF_h1_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc1(S).detach(), self.fc1(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        I_in_fb1 = surrogate.eP_switch.apply(self.fb1(out_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_rec = surrogate.eP_switch.apply(self.rec(h1_S).detach(), self.rec(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1.detach(), eLIF_h2_S1.detach(), h1_S * 1 + eLIF_h2_S2_prev.detach() * 1)
        eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2.detach(), eLIF_h2_S2.detach(), eLIF_h2_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc2(h1_S).detach(), self.fc2(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        I_in_fb2 = surrogate.eP_switch.apply(self.fb2(out_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_skip = surrogate.eP_switch.apply(self.skip(S).detach(), self.skip(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1.detach(), eLIF_h3_S1.detach(), h2_S * 1 + eLIF_h3_S2_prev.detach() * 1)
        eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2.detach(), eLIF_h3_S2.detach(), eLIF_h3_S1_prev.detach() * 1)
        I_in = surrogate.eP_switch.apply(self.fc3(h2_S).detach(), self.fc3(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        I_in_rec = surrogate.eP_switch.apply(self.rec2rec(h3_S).detach(), self.rec2rec(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        I_in_fb3 = surrogate.eP_switch.apply(self.fb3(out_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
        eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
        I_in = surrogate.eP_switch.apply(self.rec2fc(h3_S).detach(), self.rec2fc(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        I_in_skip = surrogate.eP_switch.apply(self.rec_skip(h2_S).detach(), self.rec_skip(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        eLIF_h5_U1, eLIF_h5_S1 = self.eLIF_5_1(eLIF_h5_U1, eLIF_h5_S1, out_S * 1 + eLIF_h5_S2_prev * 1)
        eLIF_h5_U2, eLIF_h5_S2 = self.eLIF_5_2(eLIF_h5_U2, eLIF_h5_S2, eLIF_h5_S1_prev * 1)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        self.spike_records['LIF'].append(S.detach().cpu().clone())
        self.spike_records['LIF1'].append(h1_S.detach().cpu().clone())
        self.spike_records['LIF2'].append(h2_S.detach().cpu().clone())
        self.spike_records['LIF3'].append(h3_S.detach().cpu().clone())
        self.spike_records['LIF_out'].append(out_S.detach().cpu().clone())
        self.spike_records['Acc_mem'].append(acc_mem.detach().cpu().clone())


        return acc_mem / new_num_steps
