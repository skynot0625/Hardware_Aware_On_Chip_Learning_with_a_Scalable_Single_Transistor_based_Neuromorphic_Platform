"""Regression checks against the exported original surrogate and neuron APIs."""

import torch

from models import model_current_eLIFE, surrogate


def main() -> None:
    torch.set_num_threads(2)
    voltage = torch.tensor([0.2, 1.2, 0.8], requires_grad=True)
    current = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
    spikes = surrogate.HeavisideCurrentBoxcarCall()(voltage, current)
    torch.testing.assert_close(spikes, torch.tensor([0., 1., 0.]))
    spikes.sum().backward()
    torch.testing.assert_close(voltage.grad, torch.tensor([1., 0., 1.]))
    assert current.grad is None
    voltage = torch.tensor([0.2, 1.2, 0.8], requires_grad=True)
    surrogate.HeavisideBoxcarCall(subthresh=1.0)(voltage).sum().backward()
    torch.testing.assert_close(voltage.grad, torch.ones(3))

    node = model_current_eLIFE.eLIF_Node_SOLOplus(
        0.5, surrogate.Heaviside_eLIFCall())
    potential, spike = node(torch.tensor([0.8]), torch.zeros(1), torch.tensor([0.3]))
    torch.testing.assert_close(potential, torch.tensor([0.7]))
    torch.testing.assert_close(spike, torch.zeros(1))
    potential, _ = node(torch.tensor([0.8]), torch.ones(1), torch.tensor([0.3]))
    torch.testing.assert_close(potential, torch.tensor([0.3]))
    node = model_current_eLIFE.eLIF_Node_SOLOplus(1.0, surrogate.Heaviside_eLIFCall())
    potential, spike = node(torch.tensor([0.8]), torch.zeros(1), torch.tensor([0.3]))
    torch.testing.assert_close(potential, torch.tensor([1.1]))
    torch.testing.assert_close(spike, torch.ones(1))
    main_spike = torch.zeros(1, requires_grad=True)
    eligible = surrogate.eLIF_fc.apply(torch.ones(1), torch.zeros(1), main_spike)
    torch.testing.assert_close(eligible, torch.ones(1))
    eligible.sum().backward()
    torch.testing.assert_close(main_spike.grad, torch.ones(1))
    v = torch.tensor([1.2], requires_grad=True)
    surrogate.Heaviside_eLIFCall()(v).sum().backward()
    torch.testing.assert_close(v.grad, torch.ones(1))
    layer = torch.nn.Linear(2, 1, bias=False)
    spike = torch.tensor([[0., 1.]], requires_grad=True)
    eligible = spike + torch.tensor([[1., 0.]])
    result = surrogate.cTraceSwitch.apply(layer(spike).detach(), layer(eligible))
    torch.testing.assert_close(result, layer(spike))
    result.sum().backward()
    torch.testing.assert_close(layer.weight.grad, eligible.detach())
    print("PASS original SG/cSG gradients, clone leak/reset/accumulation, OR and eligibility gradients")


if __name__ == "__main__":
    main()
