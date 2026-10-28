import torch
import torch.nn as nn

class Boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, subthresh):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.subthresh
        return grad_input * temp.float(), None, None

class HeavisideBoxcarCall(nn.Module):
    def __init__(self, thresh=1.0, subthresh=0.5, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
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

class eP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eP, clamp_value):
        ctx.set_materialize_grads(False)
        return torch.clamp(eP, max=clamp_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class eP_switch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_spk_mul_w, input_spk_b_mul_w):
        ctx.set_materialize_grads(False)
        return input_spk_mul_w.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return None, grad_input

class CurrentBoxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Iin, thresh, subthresh):
        ctx.save_for_backward(Iin)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return x.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (Iin,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(Iin - ctx.thresh) < ctx.subthresh
        return grad_input * temp.float(), None, None, None

class HeavisideCurrentBoxcarCall(nn.Module):
    def __init__(self, thresh=1.0, subthresh=1.0, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
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
