# Source-preserving export; see SOURCE_PROVENANCE.json and export_originals.py.
# noqa: SIZE_OK - original research classes retained verbatim for review.
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from . import surrogate


class LIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        self.LIF_decay = LIF_decay
        # tLIF neuron model
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
        # LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class LIF_current_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideCurrentBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        # self.LIF_decay = LIF_decay
        # tLIF neuron model
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
        # LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U, I_in)
        return LIF_U, LIF_S

class htLIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, scale, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()
        # 로그정규분포 파라미터 설정
        # Convert median to log-space parameter
        
        median = 4.0  # Target median in ms
        mu = torch.log(torch.tensor(median, dtype=torch.float32))  # [1][3][7]
        sigma = 0.5  # Distribution width

        # 1. Create log-normal distribution and sample
        lognormal_dist = torch.distributions.LogNormal(loc=mu, scale=sigma)  # [1][2][3]
        samples = lognormal_dist.sample((scale,))  # Sample shape (512,) [1][2][6]

        # 2. Calculate decay values
        decay_values = torch.exp(-1 / samples)  # [1][3][5]
        decay_values = torch.logit(decay_values)
        decay_values = torch.zeros_like(samples)
        self.LIF_decay = decay_values
        self.LIF_decay = nn.Parameter(decay_values)
        # self.LIF_decay = decay_values
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        # LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class htLIF_current_Node_SOLOplus(nn.Module):
    def __init__(
        self, scale, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideCurrentBoxcarCall()
    ):
        super().__init__()
        # 로그정규분포 파라미터 설정
        # Convert median to log-space parameter
        
        median = 4.0  # Target median in ms
        mu = torch.log(torch.tensor(median, dtype=torch.float32))  # [1][3][7]
        sigma = 0.5  # Distribution width
        sigma = 1.0  # Distribution width

        # 1. Create log-normal distribution and sample
        lognormal_dist = torch.distributions.LogNormal(loc=mu, scale=sigma)  # [1][2][3]
        samples = lognormal_dist.sample((scale,))  # Sample shape (512,) [1][2][6]

        # 2. Calculate decay values
        decay_values = torch.exp(-1 / samples)  # [1][3][5]
        decay_values = torch.ones_like(samples) * 0.78
        decay_values = torch.logit(decay_values)
        decay_values = torch.zeros_like(samples)
        # decay_values = torch.tensor(0, dtype=torch.float32)
        self.LIF_decay = decay_values
        self.LIF_decay = nn.Parameter(decay_values)
        # self.LIF_decay = decay_values
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        clamped_decay = torch.sigmoid(self.LIF_decay)
        # LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_S = self.surrogate_function(LIF_U, I_in)
        return LIF_U, LIF_S

class cTRACE_Node_SOLOplus(nn.Module):
    def __init__(
        self,
        TRACE_decay: torch.Tensor,
        TRACE_a_in: torch.Tensor,
        TRACE_clamp_ceiling: torch.Tensor,
        # TRACE_threshold_in: torch.Tensor,
        # TRACE_a_in: torch.Tensor,
    ):
        super().__init__()
        self.TRACE_decay = TRACE_decay
        self.TRACE_a_in = TRACE_a_in
        self.TRACE_clamp_ceiling = TRACE_clamp_ceiling

    def forward(self, TRACE_U: torch.Tensor, S_in: torch.Tensor):
        TRACE_U = self.TRACE_decay * TRACE_U + self.TRACE_a_in * S_in
        # TRACE_U = torch.clamp(TRACE_U, 0, self.TRACE_clamp_ceiling)
        return TRACE_U

class eLIF_Node_SOLOplus(nn.Module):
    def __init__(
        self, LIF_decay: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall()
    ):
        super().__init__()
        LIF_decay = torch.tensor(LIF_decay, dtype=torch.float32)
        self.LIF_decay = LIF_decay
        # tLIF neuron model
        # self.LIF_decay = nn.Parameter(LIF_decay)
        # self.LIF_decay = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        # self.a = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.surrogate_function = surrogate_function

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_before: torch.Tensor,
        I_in: torch.Tensor,
    ):
        # clamped_decay = torch.sigmoid(self.LIF_decay)
        # clampled_a = torch.sigmoid(self.a)
        # LIF_U = clamped_decay * LIF_U * (1 - S_before) + I_in
        LIF_U = self.LIF_decay * LIF_U * (1 - S_before) + I_in
        # LIF_U = I_in
        LIF_S = self.surrogate_function(LIF_U)
        return LIF_U, LIF_S

class mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_TRACE_decay: float = 0.78,
        init_TRACE_a: float = 1,
        init_TRACE_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
        rpu_config = None,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = 1024
        self.subthresh = subthresh
        self.min_steps = min_steps
        # self.ethreshold = init_TRACE_threshold

        self.time_resolution = time_resolution
        self.ethreshold = init_TRACE_decay**time_resolution
        self.ethreshold1 = init_TRACE_decay**time_resolution
        self.ethreshold2 = init_TRACE_decay**time_resolution
        self.ethreshold3 = init_TRACE_decay**time_resolution
        self.ethreshold4 = init_TRACE_decay**time_resolution

        self.cTRACE = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_1 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF1 = htLIF_current_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_2 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF2 = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_3 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF3 = htLIF_current_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_4 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF_rec = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        
        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_current_Node_SOLOplus(
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
        
        # self.fb = nn.Linear(1024, scale, bias=False)
        self.fb = nn.Linear(1024, self.scale, bias=False)
        self.LIF_fb = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        
        self.fb0 = nn.Linear(1024, self.scale, bias=False)
        self.fb1 = nn.Linear(1024, self.scale, bias=False)
        self.fb2 = nn.Linear(1024, self.scale, bias=False)
        self.fb3 = nn.Linear(1024, 1024, bias=False)

        self.skip = nn.Linear(self.scale, self.scale, bias=False)

        self.rec = nn.Linear(self.scale, self.scale, bias=False)
        

        self.outw = nn.Linear(self.scale, 200, bias=False)

        
        # parametrize.register_parametrization(self.fb0, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb1, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb2, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb3, "weight", NegativeParametrization())
        
        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eU = eS = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_eU = h1_eS = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_eU = h2_eS = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_eU = h3_eS = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        h4_eU = h4_eS = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)
        
        fb_U = fb_S = torch.zeros(batch_size, 1024, device=self.device)
        # rec_eU = rec_eS = torch.zeros(batch_size, 1024, device=self.device)
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
        eLIF_fb_U1 = eLIF_fb_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_U2 = eLIF_fb_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_S1_prev = eLIF_fb_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)
        fb_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]
            
            # init_step = random.sample(range(0, self.num_steps - self.min_steps + 1), 1)[0]
            # last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():
                # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), self.fb(h3_S))
                # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), self.fb(out_S))
                # eLIF_fb_U1, eLIF_fb_S1 = self.eLIF_fb_1(eLIF_fb_U1, eLIF_fb_S1, fb_S * 1 + eLIF_fb_S2_prev * 1)
                # eLIF_fb_U2, eLIF_fb_S2 = self.eLIF_fb_2(eLIF_fb_U2, eLIF_fb_S2, eLIF_fb_S1_prev * 1)
                # eLIF_fb_S1_prev = eLIF_fb_S1
                # eLIF_fb_S2_prev = eLIF_fb_S2            
                
                eU = self.cTRACE(eU.detach(), x[step])
                # U, S = self.LIF(U, S, self.fc(x[step]))
                # U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(fb_S))
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))
                # U, S = self.LIF(U, S, self.fc(x[step]))

                h1_eU = self.cTRACE_1(h1_eU.detach(), S)
                eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1, eLIF_h1_S1, S * 1 + eLIF_h1_S2_prev * 1)
                eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2, eLIF_h1_S2, eLIF_h1_S1_prev * 1)
                eLIF_h1_S1_prev = eLIF_h1_S1
                eLIF_h1_S2_prev = eLIF_h1_S2
                # h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(fb_S))
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
                eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1, eLIF_h2_S1, h1_S * 1 + eLIF_h2_S2_prev * 1)
                eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2, eLIF_h2_S2, eLIF_h2_S1_prev * 1)
                eLIF_h2_S1_prev = eLIF_h2_S1 
                eLIF_h2_S2_prev = eLIF_h2_S2
                
                # h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(fb_S))
                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
                eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1, eLIF_h3_S1, h2_S * 1 + eLIF_h3_S2_prev * 1)
                eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2, eLIF_h3_S2, eLIF_h3_S1_prev * 1)
                eLIF_h3_S1_prev = eLIF_h3_S1
                eLIF_h3_S2_prev = eLIF_h3_S2
                
                # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S))
                # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(fb_S))
                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
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

        # I_in = surrogate.cTraceSwitch.apply(self.fb(out_S).detach(), self.fb(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # I_in = surrogate.cTraceSwitch.apply(self.fb(h3_S).detach(), self.fb(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))     
        # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), I_in)
        
        # eLIF_fb_U1, eLIF_fb_S1 = self.eLIF_fb_1(eLIF_fb_U1, eLIF_fb_S1, fb_S * 1 + eLIF_fb_S2_prev * 1)
        # eLIF_fb_U2, eLIF_fb_S2 = self.eLIF_fb_2(eLIF_fb_U2, eLIF_fb_S2, eLIF_fb_S1_prev * 1)
        
        eU = self.cTRACE(eU.detach(), x[last_step])
        # float value? or eS?
        I_in = surrogate.cTraceSwitch.apply(self.fc(x[last_step]).detach(), self.fc(eU))
        # I_in_fb0 = surrogate.cTraceSwitch.apply(self.fb0(fb_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb0 = surrogate.cTraceSwitch.apply(self.fb0(out_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # eS = surrogate.cTrace2eSpike.apply(eU, self.ethreshold)
        # I_in = surrogate.cTraceSwitch.apply(self.fc(x[last_step]).detach(), self.fc(eS))
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)
        # U, S = self.LIF(U.detach(), S.detach(), I_in)

        h1_eU = self.cTRACE_1(h1_eU.detach(), S)
        eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1.detach(), eLIF_h1_S1.detach(), S * 1 + eLIF_h1_S2_prev.detach() * 1)
        eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2.detach(), eLIF_h1_S2.detach(), eLIF_h1_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc1(S).detach(), self.fc1(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        # I_in_fb1 = surrogate.cTraceSwitch.apply(self.fb1(fb_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb1 = surrogate.cTraceSwitch.apply(self.fb1(out_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_rec = surrogate.cTraceSwitch.apply(self.rec(h1_S).detach(), self.rec(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)
        # h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1)

        h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
        eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1.detach(), eLIF_h2_S1.detach(), h1_S * 1 + eLIF_h2_S2_prev.detach() * 1)
        eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2.detach(), eLIF_h2_S2.detach(), eLIF_h2_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc2(h1_S).detach(), self.fc2(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        # I_in_fb2 = surrogate.cTraceSwitch.apply(self.fb2(fb_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb2 = surrogate.cTraceSwitch.apply(self.fb2(out_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_skip = surrogate.cTraceSwitch.apply(self.skip(S).detach(), self.skip(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
        eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1.detach(), eLIF_h3_S1.detach(), h2_S * 1 + eLIF_h3_S2_prev.detach() * 1)
        eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2.detach(), eLIF_h3_S2.detach(), eLIF_h3_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc3(h2_S).detach(), self.fc3(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        I_in_rec = surrogate.cTraceSwitch.apply(self.rec2rec(h3_S).detach(), self.rec2rec(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        # I_in_fb3 = surrogate.cTraceSwitch.apply(self.fb3(fb_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb3 = surrogate.cTraceSwitch.apply(self.fb3(out_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
        eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
        eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
        I_in = surrogate.cTraceSwitch.apply(self.rec2fc(h3_S).detach(), self.rec2fc(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))    
        I_in_skip = surrogate.cTraceSwitch.apply(self.rec_skip(h2_S).detach(), self.rec_skip(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)
        
        eLIF_h5_U1, eLIF_h5_S1 = self.eLIF_5_1(eLIF_h5_U1, eLIF_h5_S1, out_S * 1 + eLIF_h5_S2_prev * 1)
        eLIF_h5_U2, eLIF_h5_S2 = self.eLIF_5_2(eLIF_h5_U2, eLIF_h5_S2, eLIF_h5_S1_prev * 1)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps

class mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_TRACE_decay: float = 0.78,
        init_TRACE_a: float = 1,
        init_TRACE_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps
        # self.ethreshold = init_TRACE_threshold

        self.time_resolution = time_resolution
        self.ethreshold = init_TRACE_decay**time_resolution
        self.ethreshold1 = init_TRACE_decay**time_resolution
        self.ethreshold2 = init_TRACE_decay**time_resolution
        self.ethreshold3 = init_TRACE_decay**time_resolution
        self.ethreshold4 = init_TRACE_decay**time_resolution

        self.cTRACE = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_1 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_2 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_3 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_4 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF_rec = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        
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
        
        # self.fb = nn.Linear(1024, scale, bias=False)
        self.fb = nn.Linear(1024, self.scale, bias=False)
        self.LIF_fb = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        
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
        init.xavier_uniform_(self.fb.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)

        # parametrize.register_parametrization(self.fb1, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb2, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb3, "weight", NegativeParametrization())
        
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
        # spike_recording = []
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eU = eS = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_eU = h1_eS = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_eU = h2_eS = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_eU = h3_eS = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        h4_eU = h4_eS = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)
        
        fb_U = fb_S = torch.zeros(batch_size, 1024, device=self.device)
        # rec_eU = rec_eS = torch.zeros(batch_size, 1024, device=self.device)
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
        eLIF_fb_U1 = eLIF_fb_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_U2 = eLIF_fb_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_S1_prev = eLIF_fb_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)
        fb_U.fill_(0.5)

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
                # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), self.fb(h3_S))
                # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), self.fb(out_S))
                # eLIF_fb_U1, eLIF_fb_S1 = self.eLIF_fb_1(eLIF_fb_U1, eLIF_fb_S1, fb_S * 1 + eLIF_fb_S2_prev * 1)
                # eLIF_fb_U2, eLIF_fb_S2 = self.eLIF_fb_2(eLIF_fb_U2, eLIF_fb_S2, eLIF_fb_S1_prev * 1)
                # eLIF_fb_S1_prev = eLIF_fb_S1
                # eLIF_fb_S2_prev = eLIF_fb_S2            
                
                eU = self.cTRACE(eU.detach(), x[step])
                # U, S = self.LIF(U, S, self.fc(x[step]))
                # U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(fb_S))
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))
                # U, S = self.LIF(U, S, self.fc(x[step]))

                h1_eU = self.cTRACE_1(h1_eU.detach(), S)
                eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1, eLIF_h1_S1, S * 1 + eLIF_h1_S2_prev * 1)
                eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2, eLIF_h1_S2, eLIF_h1_S1_prev * 1)
                eLIF_h1_S1_prev = eLIF_h1_S1
                eLIF_h1_S2_prev = eLIF_h1_S2
                # h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(fb_S))
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
                eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1, eLIF_h2_S1, h1_S * 1 + eLIF_h2_S2_prev * 1)
                eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2, eLIF_h2_S2, eLIF_h2_S1_prev * 1)
                eLIF_h2_S1_prev = eLIF_h2_S1 
                eLIF_h2_S2_prev = eLIF_h2_S2
                
                # h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(fb_S))
                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
                eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1, eLIF_h3_S1, h2_S * 1 + eLIF_h3_S2_prev * 1)
                eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2, eLIF_h3_S2, eLIF_h3_S1_prev * 1)
                eLIF_h3_S1_prev = eLIF_h3_S1
                eLIF_h3_S2_prev = eLIF_h3_S2
                
                # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S))
                # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(fb_S))
                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
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
                # print(step)

        # I_in = surrogate.cTraceSwitch.apply(self.fb(out_S).detach(), self.fb(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # I_in = surrogate.cTraceSwitch.apply(self.fb(h3_S).detach(), self.fb(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))     
        # fb_U, fb_S = self.LIF_fb(fb_U.detach(), fb_S.detach(), I_in)
        
        # eLIF_fb_U1, eLIF_fb_S1 = self.eLIF_fb_1(eLIF_fb_U1, eLIF_fb_S1, fb_S * 1 + eLIF_fb_S2_prev * 1)
        # eLIF_fb_U2, eLIF_fb_S2 = self.eLIF_fb_2(eLIF_fb_U2, eLIF_fb_S2, eLIF_fb_S1_prev * 1)
        
        eU = self.cTRACE(eU.detach(), x[last_step])
        # float value? or eS?
        I_in = surrogate.cTraceSwitch.apply(self.fc(x[last_step]).detach(), self.fc(eU))
        # I_in_fb0 = surrogate.cTraceSwitch.apply(self.fb0(fb_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb0 = surrogate.cTraceSwitch.apply(self.fb0(out_S).detach(), self.fb0(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # eS = surrogate.cTrace2eSpike.apply(eU, self.ethreshold)
        # I_in = surrogate.cTraceSwitch.apply(self.fc(x[last_step]).detach(), self.fc(eS))
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)
        # U, S = self.LIF(U.detach(), S.detach(), I_in)

        h1_eU = self.cTRACE_1(h1_eU.detach(), S)
        eLIF_h1_U1, eLIF_h1_S1 = self.eLIF_1_1(eLIF_h1_U1.detach(), eLIF_h1_S1.detach(), S * 1 + eLIF_h1_S2_prev.detach() * 1)
        eLIF_h1_U2, eLIF_h1_S2 = self.eLIF_1_2(eLIF_h1_U2.detach(), eLIF_h1_S2.detach(), eLIF_h1_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc1(S).detach(), self.fc1(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        # I_in_fb1 = surrogate.cTraceSwitch.apply(self.fb1(fb_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb1 = surrogate.cTraceSwitch.apply(self.fb1(out_S).detach(), self.fb1(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_rec = surrogate.cTraceSwitch.apply(self.rec(h1_S).detach(), self.rec(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)
        # h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1)

        h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
        eLIF_h2_U1, eLIF_h2_S1 = self.eLIF_2_1(eLIF_h2_U1.detach(), eLIF_h2_S1.detach(), h1_S * 1 + eLIF_h2_S2_prev.detach() * 1)
        eLIF_h2_U2, eLIF_h2_S2 = self.eLIF_2_2(eLIF_h2_U2.detach(), eLIF_h2_S2.detach(), eLIF_h2_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc2(h1_S).detach(), self.fc2(surrogate.eLIF_fc.apply(eLIF_h2_S1, eLIF_h2_S2, h1_S)))
        # I_in_fb2 = surrogate.cTraceSwitch.apply(self.fb2(fb_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb2 = surrogate.cTraceSwitch.apply(self.fb2(out_S).detach(), self.fb2(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        I_in_skip = surrogate.cTraceSwitch.apply(self.skip(S).detach(), self.skip(surrogate.eLIF_fc.apply(eLIF_h1_S1, eLIF_h1_S2, S)))
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
        eLIF_h3_U1, eLIF_h3_S1 = self.eLIF_3_1(eLIF_h3_U1.detach(), eLIF_h3_S1.detach(), h2_S * 1 + eLIF_h3_S2_prev.detach() * 1)
        eLIF_h3_U2, eLIF_h3_S2 = self.eLIF_3_2(eLIF_h3_U2.detach(), eLIF_h3_S2.detach(), eLIF_h3_S1_prev.detach() * 1)
        I_in = surrogate.cTraceSwitch.apply(self.fc3(h2_S).detach(), self.fc3(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
        I_in_rec = surrogate.cTraceSwitch.apply(self.rec2rec(h3_S).detach(), self.rec2rec(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))
        # I_in_fb3 = surrogate.cTraceSwitch.apply(self.fb3(fb_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_fb_S1, eLIF_fb_S2, fb_S)))
        I_in_fb3 = surrogate.cTraceSwitch.apply(self.fb3(out_S).detach(), self.fb3(surrogate.eLIF_fc.apply(eLIF_h5_S1, eLIF_h5_S2, out_S)))
        # h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
        eLIF_h4_U1, eLIF_h4_S1 = self.eLIF_4_1(eLIF_h4_U1, eLIF_h4_S1, h3_S * 1 + eLIF_h4_S2_prev * 1)
        eLIF_h4_U2, eLIF_h4_S2 = self.eLIF_4_2(eLIF_h4_U2, eLIF_h4_S2, eLIF_h4_S1_prev * 1)
        I_in = surrogate.cTraceSwitch.apply(self.rec2fc(h3_S).detach(), self.rec2fc(surrogate.eLIF_fc.apply(eLIF_h4_S1, eLIF_h4_S2, h3_S)))    
        I_in_skip = surrogate.cTraceSwitch.apply(self.rec_skip(h2_S).detach(), self.rec_skip(surrogate.eLIF_fc.apply(eLIF_h3_S1, eLIF_h3_S2, h2_S)))
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
        
        # print('final')

        return acc_mem / new_num_steps

class mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_pure(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_TRACE_decay: float = 0.78,
        init_TRACE_a: float = 1,
        init_TRACE_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps
        # self.ethreshold = init_TRACE_threshold

        self.time_resolution = time_resolution
        self.ethreshold = init_TRACE_decay**time_resolution
        self.ethreshold1 = init_TRACE_decay**time_resolution
        self.ethreshold2 = init_TRACE_decay**time_resolution
        self.ethreshold3 = init_TRACE_decay**time_resolution
        self.ethreshold4 = init_TRACE_decay**time_resolution

        self.cTRACE = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_1 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF1 = htLIF_current_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_2 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF2 = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_3 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF3 = htLIF_current_Node_SOLOplus(
            1024,
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_4 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF_rec = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        
        self.rec2fc = nn.Linear(1024, 1024, bias=False)
        self.LIF_out = LIF_current_Node_SOLOplus(
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
        
        # self.fb = nn.Linear(1024, scale, bias=False)
        self.fb = nn.Linear(1024, self.scale, bias=False)
        self.LIF_fb = LIF_current_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideCurrentBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        
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
        init.xavier_uniform_(self.fb.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)
        # parametrize.register_parametrization(self.fb1, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb2, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb3, "weight", NegativeParametrization())
        
        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eU = eS = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_eU = h1_eS = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_eU = h2_eS = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_eU = h3_eS = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        h4_eU = h4_eS = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)
        
        fb_U = fb_S = torch.zeros(batch_size, 1024, device=self.device)
        # rec_eU = rec_eS = torch.zeros(batch_size, 1024, device=self.device)
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
        eLIF_fb_U1 = eLIF_fb_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_U2 = eLIF_fb_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_S1_prev = eLIF_fb_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)
        fb_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():         
                
                eU = self.cTRACE(eU.detach(), x[step])
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                h1_eU = self.cTRACE_1(h1_eU.detach(), S)
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))

                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1
  
        eU = self.cTRACE(eU.detach(), x[last_step])
        # float value? or eS?
        I_in = self.fc(x[last_step])
        I_in_fb0 = self.fb0(out_S)
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        h1_eU = self.cTRACE_1(h1_eU.detach(), S)
        I_in = self.fc1(S)
        I_in_fb1 = self.fb1(out_S)
        I_in_rec = self.rec(h1_S)
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
        I_in = self.fc2(h1_S)
        I_in_fb2 = self.fb2(out_S)
        I_in_skip = self.skip(S)
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
        I_in = self.fc3(h2_S)
        I_in_rec = self.rec2rec(h3_S)
        I_in_fb3 = self.fb3(out_S)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
        I_in = self.rec2fc(h3_S)    
        I_in_skip = self.rec_skip(h2_S)
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps

class mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_pure(nn.Module):
    def __init__(
        self,
        scale=1024,
        num_steps: int = 28,
        time_resolution=4,
        init_LIF_decay: float = 0.78,  # membrane decaying time constant
        init_TRACE_decay: float = 0.78,
        init_TRACE_a: float = 1,
        init_TRACE_clamp_ceiling: float = 100,
        subthresh=1.0,
        min_steps: int = 4,
    ):
        super().__init__()
        self.num_steps = 20
        self.scale = scale
        self.subthresh = subthresh
        self.min_steps = min_steps
        # self.ethreshold = init_TRACE_threshold

        self.time_resolution = time_resolution
        self.ethreshold = init_TRACE_decay**time_resolution
        self.ethreshold1 = init_TRACE_decay**time_resolution
        self.ethreshold2 = init_TRACE_decay**time_resolution
        self.ethreshold3 = init_TRACE_decay**time_resolution
        self.ethreshold4 = init_TRACE_decay**time_resolution

        self.cTRACE = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
        )
        self.fc = nn.Linear(700, self.scale, bias=False)
        self.LIF = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )

        self.cTRACE_1 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_2 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_3 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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

        self.cTRACE_4 = cTRACE_Node_SOLOplus(
            TRACE_decay=init_TRACE_decay,
            TRACE_a_in=init_TRACE_a,
            TRACE_clamp_ceiling=init_TRACE_clamp_ceiling,
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
        self.LIF_rec = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        
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
        
        # self.fb = nn.Linear(1024, scale, bias=False)
        self.fb = nn.Linear(1024, self.scale, bias=False)
        self.LIF_fb = LIF_Node_SOLOplus(
            LIF_decay=init_LIF_decay,
            surrogate_function=surrogate.HeavisideBoxcarCall(
                thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_1 = eLIF_Node_SOLOplus(
            LIF_decay=0.5,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        self.eLIF_fb_2 = eLIF_Node_SOLOplus(
            LIF_decay=1,
            surrogate_function=surrogate.Heaviside_eLIFCall(
                thresh=1.0, alpha=1.0, spiking=True
            ),
        )
        
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
        init.xavier_uniform_(self.fb.weight)
        init.xavier_uniform_(self.fb0.weight)
        init.xavier_uniform_(self.fb1.weight)
        init.xavier_uniform_(self.fb2.weight)
        init.xavier_uniform_(self.fb3.weight)
        init.xavier_uniform_(self.skip.weight)
        init.xavier_uniform_(self.rec.weight)

        self.outw = nn.Linear(self.scale, 200, bias=False)
        init.xavier_uniform_(self.outw.weight)
        # parametrize.register_parametrization(self.fb1, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb2, "weight", NegativeParametrization())
        # parametrize.register_parametrization(self.fb3, "weight", NegativeParametrization())
        
        self.boost1 = nn.AvgPool1d(10, 10)

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.squeeze(2)       # (32, 20, 1, 700) → (32, 20, 700)
        x = x.permute(1, 0, 2) # (32, 20, 700) → (20, 32, 700)

        eU = eS = torch.zeros(batch_size, 700, device=self.device)
        U = S = torch.zeros(batch_size, 1024, device=self.device)
        h1_eU = h1_eS = torch.zeros(batch_size, 1024, device=self.device)
        h1_U = h1_S = torch.zeros(batch_size, 1024, device=self.device)
        h2_eU = h2_eS = torch.zeros(batch_size, 1024, device=self.device)
        h2_U = h2_S = torch.zeros(batch_size, 1024, device=self.device)
        h3_eU = h3_eS = torch.zeros(batch_size, 1024, device=self.device)
        h3_U = h3_S = torch.zeros(batch_size, 1024, device=self.device)
        h4_eU = h4_eS = torch.zeros(batch_size, 1024, device=self.device)
        out_U = out_S = torch.zeros(batch_size, 1024, device=self.device)
        
        fb_U = fb_S = torch.zeros(batch_size, 1024, device=self.device)
        # rec_eU = rec_eS = torch.zeros(batch_size, 1024, device=self.device)
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
        eLIF_fb_U1 = eLIF_fb_S1 = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_U2 = eLIF_fb_S2 = torch.zeros(batch_size, 1024, device=self.device)

        eLIF_h1_S1_prev = eLIF_h1_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h2_S1_prev = eLIF_h2_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h3_S1_prev = eLIF_h3_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h4_S1_prev = eLIF_h4_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_h5_S1_prev = eLIF_h5_S2_prev = torch.zeros(batch_size, 1024, device=self.device)
        eLIF_fb_S1_prev = eLIF_fb_S2_prev = torch.zeros(batch_size, 1024, device=self.device)

        U.fill_(0.5)
        h1_U.fill_(0.5)
        h2_U.fill_(0.5)
        h3_U.fill_(0.5)
        out_U.fill_(0.5)
        fb_U.fill_(0.5)

        init_step = 0
        last_step = self.num_steps - 1
        new_num_steps = self.num_steps

        if self.training:
            init_step = 0
            last_step = random.sample(range(init_step + self.min_steps - 1, self.num_steps), 1)[0]

            new_num_steps = last_step - init_step + 1

        for step in range(init_step, last_step - 1):
            with torch.no_grad():

                eU = self.cTRACE(eU.detach(), x[step])
                U, S = self.LIF(U, S, self.fc(x[step]) + self.fb0(out_S))

                h1_eU = self.cTRACE_1(h1_eU.detach(), S)
                h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), self.fc1(S) + self.fb1(out_S) + self.rec(h1_S))

                h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
                h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), self.fc2(h1_S) + self.fb2(out_S) + self.skip(S))

                h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
                h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), self.fc3(h2_S) + self.rec2rec(h3_S) + self.fb3(out_S))

                h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
                out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), self.rec2fc(h3_S) + self.rec_skip(h2_S))
                
                boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

                acc_mem += boost1

        eU = self.cTRACE(eU.detach(), x[last_step])
        # float value? or eS?
        I_in = self.fc(x[last_step])
        I_in_fb0 = self.fb0(out_S)
        U, S = self.LIF(U.detach(), S.detach(), I_in + I_in_fb0)

        h1_eU = self.cTRACE_1(h1_eU.detach(), S)
        I_in = self.fc1(S)
        I_in_fb1 = self.fb1(out_S)
        I_in_rec = self.rec(h1_S)
        h1_U, h1_S = self.LIF1(h1_U.detach(), h1_S.detach(), I_in + I_in_fb1 + I_in_rec)

        h2_eU = self.cTRACE_2(h2_eU.detach(), h1_S)
        I_in = self.fc2(h1_S)
        I_in_fb2 = self.fb2(out_S)
        I_in_skip = self.skip(S)
        h2_U, h2_S = self.LIF2(h2_U.detach(), h2_S.detach(), I_in + I_in_fb2 + I_in_skip)

        h3_eU = self.cTRACE_3(h3_eU.detach(), h2_S)
        I_in = self.fc3(h2_S)
        I_in_rec = self.rec2rec(h3_S)
        I_in_fb3 = self.fb3(out_S)
        h3_U, h3_S = self.LIF3(h3_U.detach(), h3_S.detach(), I_in + I_in_rec + I_in_fb3)

        h4_eU = self.cTRACE_4(h4_eU.detach(), h3_S)
        I_in = self.rec2fc(h3_S)
        I_in_skip = self.rec_skip(h2_S)
        out_U, out_S = self.LIF_out(out_U.detach(), out_S.detach(), I_in + I_in_skip)

        boost1 = self.boost1(self.outw(out_S).unsqueeze(1)).squeeze(1)

        acc_mem += boost1

        return acc_mem / new_num_steps
