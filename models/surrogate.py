import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, subthresh):
        # spike threshold, Heaviside
        # store membrane potential before reset
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, BoxCar
        # stored membrane potential before reset
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.subthresh
        # return grad_input, None, None
        # return grad_input * temp.float(), None, None
        # return grad_input * temp.float() / (ctx.subthresh*2), None, None
        return grad_input * temp.float(), None, None


class HeavisideBoxcarCall(nn.Module):
    def __init__(self, thresh=1.0, subthresh=0.5, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        # FIX: 강제 디바이스 할당(.to("cuda")) 제거 및 register_buffer 사용
        self.register_buffer("thresh", torch.tensor(thresh))
        self.register_buffer("subthresh", torch.tensor(subthresh))
        if spiking:
            self.f = Boxcar.apply
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.thresh, self.subthresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return x * alpha

class cTRACEClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eU, clamp_value):
        ctx.set_materialize_grads(False)  
        return torch.clamp(eU, max=clamp_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class cTraceSwitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_spk_mul_w, input_spk_b_mul_w):
        ctx.set_materialize_grads(False)
        return input_spk_mul_w.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return None, grad_input

class cTRACE2eSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eU, ethreshold):
        ctx.set_materialize_grads(False)
        eS = eU.gt(ethreshold).float()
        return eS.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class cTRACESwitch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_spk_mul_w, input_spk_b_mul_w):
        ctx.set_materialize_grads(False)
        return input_spk_mul_w.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return None, grad_input


class AlwaysOnbeyondAcc(torch.autograd.Function):
    # Define approximate firing function
    # 항상 모든 step에서 output에서는 grad가 만들어지게 유도, acc_nueron에 들어오는 spk가 0이여도 작동
    # always update - 상관없을 듯, 어자피 spike 나오면, Vmem이 0이므로 update 안됨.
    @staticmethod
    def forward(ctx, input_spk):
        ctx.set_materialize_grads(False)
        return input_spk

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class CurrentBoxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Iin, thresh, subthresh):
        # spike threshold, Heaviside
        # store membrane potential before reset
        ctx.save_for_backward(Iin)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return x.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, BoxCar
        # stored membrane potential before reset
        (Iin,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(Iin - ctx.thresh) < ctx.subthresh
        # print(grad_input * temp.float())
        # return grad_input, None, None
        # return grad_input * temp.float(), None, None
        return grad_input * temp.float(), None, None, None

class HeavisideCurrentBoxcarCall(nn.Module):
    def __init__(self, thresh=1.0, subthresh=1.0, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        # FIX: 강제 디바이스 할당(.to("cuda")) 제거 및 register_buffer 사용
        self.register_buffer("thresh", torch.tensor(thresh))
        self.register_buffer("subthresh", torch.tensor(subthresh))
        if spiking:
            self.f = CurrentBoxcar.apply
        else:
            self.f = self.primitive_function

    def forward(self, x, Iin):
        return self.f(x, Iin, self.thresh, self.subthresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return x * alpha
    
    
class eLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        # spike threshold, Heaviside
        # store membrane potential before reset
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Heaviside_eLIFCall(nn.Module):
    def __init__(self, thresh=1.0, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        # FIX: 강제 디바이스 할당(.to("cuda")) 제거 및 register_buffer 사용
        self.register_buffer("thresh", torch.tensor(thresh))
        if spiking:
            self.f = eLIF.apply
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.thresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return x * alpha


class eLIF_fc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eS1, eS2, S):
        input_spk = torch.maximum(torch.maximum(eS1, eS2), S)
        return input_spk

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return None, None, grad_input