# Source-preserving export; see SOURCE_PROVENANCE.json and export_originals.py.
# noqa: SIZE_OK - original research classes retained verbatim for review.
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from . import surrogate


class LIF_Node(nn.Module):
    def __init__(
        self, LIF_tau: torch.Tensor, surrogate_function=surrogate.HeavisideBoxcarCall
    ):
        super().__init__()
        self.LIF_decay = torch.sigmoid(torch.tensor(LIF_tau))
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

class cTRACE_Node(nn.Module):
    def __init__(self, cTRACE_tau: torch.Tensor, cTRACE_clamp_value):
        super().__init__()

        self.cTRACE_decay = torch.sigmoid(torch.tensor(cTRACE_tau))
        self.clamp_value = cTRACE_clamp_value

    def forward(self, mem: torch.Tensor, spk_in: torch.Tensor):
        mem =  self.cTRACE_decay * mem.detach() + spk_in
        mem = surrogate.cTRACEClamp.apply(mem, self.clamp_value)
        # spike_b = self.surrogate_function(mem)
        
        # return mem, spike_b
        return mem

class LI_Node(nn.Module):
    def __init__(
        self,
        LI_tau: torch.Tensor,
    ):
        super().__init__()
        self.LI_decay = torch.sigmoid(torch.tensor(LI_tau))

    def forward(
        self,
        LIF_U: torch.Tensor,
        S_in: torch.Tensor,
    ):
        LIF_U = self.LI_decay * LIF_U + S_in
        return LIF_U

class SOLO_PreactResNet18(nn.Module):
    def __init__(
        self,
        scale = 64, 
        num_steps: int = 5,
        init_LIF_tau: float = 1.0,
        init_LI_tau: float =1.0,
        subthresh  = 0.5,
        num_classes = 10,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.scale = scale
        self.subthresh = subthresh
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, self.scale, 3, stride=1, padding=1, bias=False)  
        #
        
        self.SRB1_bn1 = nn.BatchNorm2d(self.scale)
        self.LIF1 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_conv1 = nn.Conv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB1_bn2 = nn.BatchNorm2d(self.scale)
        self.LIF2 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_conv2 = nn.Conv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB1_skip = nn.Conv2d(self.scale, self.scale, 1, stride=1, bias=False)

        self.SRB2_bn1 = nn.BatchNorm2d(self.scale)
        self.LIF3 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_conv1 = nn.Conv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB2_bn2 = nn.BatchNorm2d(self.scale)
        self.LIF4 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_conv2 = nn.Conv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False)
        self.SRB2_skip = nn.Sequential(nn.BatchNorm2d(self.scale))   

        self.SRB3_bn1 = nn.BatchNorm2d(self.scale)
        self.LIF5 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_conv1 = nn.Conv2d(self.scale, self.scale*2, 3, stride=2, padding=1, bias=False)
        self.SRB3_bn2 = nn.BatchNorm2d(self.scale*2)
        self.LIF6 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_conv2 = nn.Conv2d(self.scale*2, self.scale*2, 3, stride=1, padding=1, bias=False)
        self.SRB3_skip = nn.Conv2d(self.scale, self.scale*2, 1, stride=2, bias=False)


        self.SRB4_bn1 = nn.BatchNorm2d(self.scale*2)
        self.LIF7 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_conv1 = nn.Conv2d(self.scale*2, self.scale*2, 3, stride=1, padding=1, bias=False)
        self.SRB4_bn2 = nn.BatchNorm2d(self.scale*2)
        self.LIF8 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_conv2 = nn.Conv2d(self.scale*2, self.scale*2 , 3, stride=1, padding=1, bias=False)
        self.SRB4_skip = nn.Sequential(nn.BatchNorm2d(self.scale*2)) 

        self.SRB5_bn1 = nn.BatchNorm2d(self.scale*2)
        self.LIF9 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_conv1 = nn.Conv2d(self.scale*2, self.scale*4, 3, stride=2, padding=1, bias=False)
        self.SRB5_bn2 = nn.BatchNorm2d(self.scale*4)
        self.LIF10 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_conv2 = nn.Conv2d(self.scale*4, self.scale*4, 3, stride=1, padding=1, bias=False)
        self.SRB5_skip = nn.Conv2d(self.scale*2, self.scale*4, 1, stride=2, bias=False)

        self.SRB6_bn1 = nn.BatchNorm2d(self.scale*4)
        self.LIF11 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_conv1 = nn.Conv2d(self.scale*4, self.scale*4, 3, stride=1, padding=1, bias=False)
        self.SRB6_bn2 = nn.BatchNorm2d(self.scale*4)
        self.LIF12 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_conv2 = nn.Conv2d(self.scale*4, self.scale*4, 3, stride=1, padding=1, bias=False)
        self.SRB6_skip = nn.Sequential(nn.BatchNorm2d(self.scale*4)) 

        self.SRB7_bn1 = nn.BatchNorm2d(self.scale*4)
        self.LIF13 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_conv1 = nn.Conv2d(self.scale*4, self.scale*8, 3, stride=2, padding=1, bias=False)
        self.SRB7_bn2 = nn.BatchNorm2d(self.scale*8)
        self.LIF14 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_conv2 = nn.Conv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False)
        self.SRB7_skip = nn.Conv2d(self.scale*4, self.scale*8, 1, stride=2, bias=False)  

        self.SRB8_bn1 = nn.BatchNorm2d(self.scale*8)
        self.LIF15 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_conv1 = nn.Conv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False)
        self.SRB8_bn2 = nn.BatchNorm2d(self.scale*8)
        self.LIF16 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_conv2 = nn.Conv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False)
        self.SRB8_skip = nn.Sequential(nn.BatchNorm2d(self.scale*8))  

        # fc
        self.bn1 = nn.BatchNorm2d(self.scale*8)
        self.LIF17 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.pool1 = nn.AvgPool2d(4)

        self.fc1 = nn.Linear(self.scale*8, num_classes*10, bias=False)
        
        self.LIF18 = LIF_Node(LIF_tau=init_LIF_tau, surrogate_function=surrogate.HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))

        self.fc2 = nn.Linear(num_classes*10, num_classes*10, bias=False)

        self.boost1 = nn.AvgPool1d(10, 10)

        self.LI = LI_Node(LI_tau=init_LI_tau)

    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_1_mem = srb1_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_2_mem = srb1_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb2_1_mem = srb2_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb2_2_mem = srb2_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb3_1_mem = srb3_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb3_2_mem = srb3_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb4_1_mem = srb4_1_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb4_2_mem = srb4_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device) 
        srb5_1_mem = srb5_1_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb5_2_mem = srb5_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb6_1_mem = srb6_1_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        srb6_2_mem = srb6_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb7_1_mem = srb7_1_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        srb7_2_mem = srb7_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device) 
        srb8_1_mem = srb8_1_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)
        srb8_2_mem = srb8_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)        
        h1_mem = h1_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device) 
        h2_mem = h2_spike = torch.zeros(batch_size, self.num_classes*10, device=self.device)
        boost1 = torch.zeros(batch_size, self.num_classes, device=self.device)
        acc_mem = torch.zeros(batch_size, self.num_classes, device=self.device)

        c1_mem.fill_(0.5)
        srb1_1_mem.fill_(0.5)
        srb1_2_mem.fill_(0.5)
        srb2_1_mem.fill_(0.5)
        srb2_2_mem.fill_(0.5)
        srb3_1_mem.fill_(0.5)
        srb3_2_mem.fill_(0.5)
        srb4_1_mem.fill_(0.5)
        srb4_2_mem.fill_(0.5)
        srb5_1_mem.fill_(0.5)
        srb5_2_mem.fill_(0.5)
        srb6_1_mem.fill_(0.5)
        srb6_2_mem.fill_(0.5)
        srb7_1_mem.fill_(0.5)
        srb7_2_mem.fill_(0.5)
        srb8_1_mem.fill_(0.5)
        srb8_2_mem.fill_(0.5)
        h1_mem.fill_(0.5)
        h2_mem.fill_(0.5)

        for step in range(self.num_steps-1):
            with torch.no_grad():
                c1_out = self.conv1(x)

                #

                srb1_1_mem, srb1_1_spike = self.LIF1(srb1_1_mem.detach(), srb1_1_spike.detach(), self.SRB1_bn1(c1_out))

                srb1_2_mem, srb1_2_spike = self.LIF2(srb1_2_mem.detach(), srb1_2_spike.detach(), self.SRB1_bn2(self.SRB1_conv1(srb1_1_spike)))
                
                srb1_out = self.SRB1_conv2(srb1_2_spike) +self.SRB1_skip(c1_out)

                srb2_1_mem, srb2_1_spike = self.LIF3(srb2_1_mem.detach(), srb2_1_spike.detach(), self.SRB2_bn1(srb1_out))

                srb2_2_mem, srb2_2_spike = self.LIF4(srb2_2_mem.detach(), srb2_2_spike.detach(), self.SRB2_bn2(self.SRB2_conv1(srb2_1_spike)))
                
                srb2_out = self.SRB2_conv2(srb2_2_spike) + self.SRB2_skip(srb1_out)

                srb3_1_mem, srb3_1_spike = self.LIF5(srb3_1_mem.detach(), srb3_1_spike.detach(), self.SRB3_bn1(srb2_out))

                srb3_2_mem, srb3_2_spike = self.LIF6(srb3_2_mem.detach(), srb3_2_spike.detach(), self.SRB3_bn2(self.SRB3_conv1(srb3_1_spike)))
                
                srb3_out = self.SRB3_conv2(srb3_2_spike) +self.SRB3_skip(srb2_out)

                srb4_1_mem, srb4_1_spike = self.LIF7(srb4_1_mem.detach(), srb4_1_spike.detach(), self.SRB4_bn1(srb3_out))

                srb4_2_mem, srb4_2_spike = self.LIF8(srb4_2_mem.detach(), srb4_2_spike.detach(), self.SRB4_bn2(self.SRB4_conv1(srb4_1_spike)))
                
                srb4_out = self.SRB4_conv2(srb4_2_spike) +self.SRB4_skip(srb3_out)

                srb5_1_mem, srb5_1_spike = self.LIF9(srb5_1_mem.detach(), srb5_1_spike.detach(), self.SRB5_bn1(srb4_out))

                srb5_2_mem, srb5_2_spike = self.LIF10(srb5_2_mem.detach(), srb5_2_spike.detach(), self.SRB5_bn2(self.SRB5_conv1(srb5_1_spike)))
                
                srb5_out = self.SRB5_conv2(srb5_2_spike) +self.SRB5_skip(srb4_out)

                srb6_1_mem, srb6_1_spike = self.LIF11(srb6_1_mem.detach(), srb6_1_spike.detach(), self.SRB6_bn1(srb5_out))

                srb6_2_mem, srb6_2_spike = self.LIF12(srb6_2_mem.detach(), srb6_2_spike.detach(), self.SRB6_bn2(self.SRB6_conv1(srb6_1_spike)))
                
                srb6_out = self.SRB6_conv2(srb6_2_spike) +self.SRB6_skip(srb5_out)

                srb7_1_mem, srb7_1_spike = self.LIF13(srb7_1_mem.detach(), srb7_1_spike.detach(), self.SRB7_bn1(srb6_out))

                srb7_2_mem, srb7_2_spike = self.LIF14(srb7_2_mem.detach(), srb7_2_spike.detach(), self.SRB7_bn2(self.SRB7_conv1(srb7_1_spike)))
                
                srb7_out = self.SRB7_conv2(srb7_2_spike) +self.SRB7_skip(srb6_out)

                srb8_1_mem, srb8_1_spike = self.LIF15(srb8_1_mem.detach(), srb8_1_spike.detach(), self.SRB8_bn1(srb7_out))

                srb8_2_mem, srb8_2_spike = self.LIF16(srb8_2_mem.detach(), srb8_2_spike.detach(), self.SRB8_bn2(self.SRB8_conv1(srb8_1_spike)))
                
                srb8_out = self.SRB8_conv2(srb8_2_spike) +self.SRB8_skip(srb7_out)

                h1_mem, h1_spike = self.LIF17(h1_mem.detach(), h1_spike.detach(),  self.bn1(srb8_out))

                h2_mem, h2_spike = self.LIF18(h2_mem.detach(), h2_spike.detach(),  self.fc1(self.pool1(h1_spike).view(batch_size, -1)))
                # h2_mem, h2_spike = self.LIF18(h2_mem.detach(), h2_spike.detach(),  self.fc1(self.pool2(self.pool1(h1_spike)).view(batch_size, -1)))

                # new
                # boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)
                boost1 = self.boost1(self.fc2(h2_spike).unsqueeze(1)).squeeze(1)

                # acc_mem = self.LI(acc_mem.detach(), boost1)
                acc_mem += boost1

        c1_out = self.conv1(x)

        #

        srb1_1_mem, srb1_1_spike = self.LIF1(srb1_1_mem.detach(), srb1_1_spike.detach(), self.SRB1_bn1(c1_out))

        srb1_2_mem, srb1_2_spike = self.LIF2(srb1_2_mem.detach(), srb1_2_spike.detach(), self.SRB1_bn2(self.SRB1_conv1(srb1_1_spike)))
        
        srb1_out = self.SRB1_conv2(srb1_2_spike) +self.SRB1_skip(c1_out)

        srb2_1_mem, srb2_1_spike = self.LIF3(srb2_1_mem.detach(), srb2_1_spike.detach(), self.SRB2_bn1(srb1_out))

        srb2_2_mem, srb2_2_spike = self.LIF4(srb2_2_mem.detach(), srb2_2_spike.detach(), self.SRB2_bn2(self.SRB2_conv1(srb2_1_spike)))
        
        srb2_out = self.SRB2_conv2(srb2_2_spike) + self.SRB2_skip(srb1_out)

        srb3_1_mem, srb3_1_spike = self.LIF5(srb3_1_mem.detach(), srb3_1_spike.detach(), self.SRB3_bn1(srb2_out))

        srb3_2_mem, srb3_2_spike = self.LIF6(srb3_2_mem.detach(), srb3_2_spike.detach(), self.SRB3_bn2(self.SRB3_conv1(srb3_1_spike)))
        
        srb3_out = self.SRB3_conv2(srb3_2_spike) +self.SRB3_skip(srb2_out)

        srb4_1_mem, srb4_1_spike = self.LIF7(srb4_1_mem.detach(), srb4_1_spike.detach(), self.SRB4_bn1(srb3_out))

        srb4_2_mem, srb4_2_spike = self.LIF8(srb4_2_mem.detach(), srb4_2_spike.detach(), self.SRB4_bn2(self.SRB4_conv1(srb4_1_spike)))
        
        srb4_out = self.SRB4_conv2(srb4_2_spike) +self.SRB4_skip(srb3_out)

        srb5_1_mem, srb5_1_spike = self.LIF9(srb5_1_mem.detach(), srb5_1_spike.detach(), self.SRB5_bn1(srb4_out))

        srb5_2_mem, srb5_2_spike = self.LIF10(srb5_2_mem.detach(), srb5_2_spike.detach(), self.SRB5_bn2(self.SRB5_conv1(srb5_1_spike)))
        
        srb5_out = self.SRB5_conv2(srb5_2_spike) +self.SRB5_skip(srb4_out)

        srb6_1_mem, srb6_1_spike = self.LIF11(srb6_1_mem.detach(), srb6_1_spike.detach(), self.SRB6_bn1(srb5_out))

        srb6_2_mem, srb6_2_spike = self.LIF12(srb6_2_mem.detach(), srb6_2_spike.detach(), self.SRB6_bn2(self.SRB6_conv1(srb6_1_spike)))
        
        srb6_out = self.SRB6_conv2(srb6_2_spike) +self.SRB6_skip(srb5_out)

        srb7_1_mem, srb7_1_spike = self.LIF13(srb7_1_mem.detach(), srb7_1_spike.detach(), self.SRB7_bn1(srb6_out))

        srb7_2_mem, srb7_2_spike = self.LIF14(srb7_2_mem.detach(), srb7_2_spike.detach(), self.SRB7_bn2(self.SRB7_conv1(srb7_1_spike)))
        
        srb7_out = self.SRB7_conv2(srb7_2_spike) +self.SRB7_skip(srb6_out)

        srb8_1_mem, srb8_1_spike = self.LIF15(srb8_1_mem.detach(), srb8_1_spike.detach(), self.SRB8_bn1(srb7_out))

        srb8_2_mem, srb8_2_spike = self.LIF16(srb8_2_mem.detach(), srb8_2_spike.detach(), self.SRB8_bn2(self.SRB8_conv1(srb8_1_spike)))
        
        srb8_out = self.SRB8_conv2(srb8_2_spike) +self.SRB8_skip(srb7_out)

        h1_mem, h1_spike = self.LIF17(h1_mem.detach(), h1_spike.detach(),  self.bn1(srb8_out))

        h2_mem, h2_spike = self.LIF18(h2_mem.detach(), h2_spike.detach(),  self.fc1(self.pool1(h1_spike).view(batch_size, -1)))
        # h2_mem, h2_spike = self.LIF18(h2_mem.detach(), h2_spike.detach(),  self.fc1(self.pool2(self.pool1(h1_spike)).view(batch_size, -1)))

        # new
        # boost1 = self.boost1(h2_spike.unsqueeze(1)).squeeze(1)
        boost1 = self.boost1(self.fc2(h2_spike).unsqueeze(1)).squeeze(1)

        # acc_mem = self.LI(acc_mem.detach(), boost1)
        acc_mem += boost1

        return acc_mem
