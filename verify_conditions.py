"""Matched four-condition checks for the review examples."""

import torch
import random

from models import PreActResNet18, SHDELife
from models.neurons import LIFState, Surrogate, lif_step


def main() -> None:
    torch.set_num_threads(2)
    gradients = []
    outputs = []
    for surrogate in Surrogate:
        current = torch.tensor([0.2], requires_grad=True)
        state = LIFState(torch.tensor([-2.0]), torch.zeros(1))
        result = lif_step(current, state, torch.tensor(1.0), surrogate)
        result.spike.sum().backward()
        outputs.append(result.spike.detach())
        gradients.append(current.grad)
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(gradients[0], torch.zeros(1))
    torch.testing.assert_close(gradients[1], torch.ones(1))
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for network in [PreActResNet18, SHDELife]:
            for elife in [False, True]:
                values = []
                for surrogate in Surrogate:
                    torch.manual_seed(42)
                    model = network(scale=32, surrogate=surrogate, elife=elife).to(device)
                    sample = (torch.randn(2, 3, 32, 32) if network is PreActResNet18 else
                              torch.poisson(torch.full((2, 20, 700), 2.0))).to(device)
                    model.eval()
                    values.append(model(sample).detach())
                    model.train()
                    random.seed(42)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    before = [p.detach().clone() for p in model.parameters()]
                    loss = (model(sample) - 1).square().mean()
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    assert all(torch.isfinite(g).all() for g in grads)
                    assert any(torch.count_nonzero(g) for g in grads)
                    optimizer.step()
                    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
                    print(f"PASS {device} {network.__name__} {surrogate.value} elife={elife}")
                torch.testing.assert_close(values[0], values[1])
    print("PASS matched forward outputs, distinct SG/cSG masks, all four optimizer paths")


if __name__ == "__main__":
    main()
