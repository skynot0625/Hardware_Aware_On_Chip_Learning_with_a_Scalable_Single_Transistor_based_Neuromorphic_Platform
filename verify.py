"""Executable mathematical and model smoke checks; no dataset download needed."""

import torch

from models import PreActResNet18, SHDELife
from models.neurons import Eligibility, current_spike, project


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    voltage = torch.tensor([0.2, 1.2, 0.8], requires_grad=True)
    current = torch.tensor([1.0, 3.0, 0.5], requires_grad=True)
    spikes = current_spike(voltage, current)
    torch.testing.assert_close(spikes, torch.tensor([0., 1., 0.]))
    spikes.sum().backward()
    torch.testing.assert_close(voltage.grad, torch.tensor([1., 0., 1.]))
    assert current.grad is None

    state = Eligibility(torch.tensor([0.8]), torch.zeros(1), torch.tensor([0.8]), torch.zeros(1))
    updated = state.advance(torch.zeros(1))
    torch.testing.assert_close(updated.voltage1, torch.tensor([0.4]))
    torch.testing.assert_close(updated.voltage2, torch.tensor([0.8]))
    state = Eligibility(torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1))
    updated = state.advance(torch.ones(1))
    torch.testing.assert_close(updated.voltage1, torch.tensor([2.]))
    torch.testing.assert_close(updated.voltage2, torch.tensor([1.]))
    main_spike = torch.zeros(1, requires_grad=True)
    eligible = updated.encode(main_spike)
    torch.testing.assert_close(eligible, torch.ones(1))
    eligible.sum().backward()
    torch.testing.assert_close(main_spike.grad, torch.ones(1))

    layer = torch.nn.Linear(2, 1, bias=False)
    spike = torch.tensor([[0., 1.]], requires_grad=True)
    eligible = spike + torch.tensor([[1., 0.]])
    value = project(layer, spike, eligible)
    torch.testing.assert_close(value, layer(spike))
    value.sum().backward()
    torch.testing.assert_close(layer.weight.grad, eligible.detach())

    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for model, sample in [
            (PreActResNet18(scale=2), torch.randn(2, 3, 32, 32)),
            (PreActResNet18(scale=2, elife=True), torch.randn(2, 3, 32, 32)),
            (SHDELife(scale=32), torch.poisson(torch.full((2, 20, 700), 2.0))),
        ]:
            model = model.to(device)
            sample = sample.to(device)
            model.eval()
            with torch.no_grad():
                expected = model(sample)
                torch.testing.assert_close(expected, model(sample))
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            before = [p.detach().clone() for p in model.parameters()]
            output = model(sample)
            loss = (output - 1).square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            optimizer.step()
            assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
            print(f"PASS {device} {type(model).__name__} elife={getattr(model, 'elife', True)} "
                  f"shape={tuple(output.shape)} loss={float(loss.detach()):.6f}")
    print("PASS cSG gradient, clone recurrence, OR gradient, eligibility weight gradient, state reset, optimizer update")


if __name__ == "__main__":
    main()
