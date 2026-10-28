"""Exercise all twelve architecture/learning-condition combinations."""

import gc
import random

import torch

from models import PreActResNet18, SHDELife


def main() -> None:
    torch.set_num_threads(2)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        for architecture in ["resnet18", "arch1", "arch2"]:
            for elife in [False, True]:
                for sg in ["sg", "csg"]:
                    torch.manual_seed(42)
                    random.seed(42)
                    if architecture == "resnet18":
                        model = PreActResNet18(scale=4, surrogate=sg, elife=elife)
                        sample = torch.randn(2, 3, 32, 32)
                    else:
                        model = SHDELife(surrogate=sg, elife=elife, architecture=architecture)
                        assert hasattr(model, "fb0") == (architecture == "arch2")
                        assert hasattr(model, "cTRACE") == elife
                        assert not hasattr(model, "cTRACE_1")
                        sample = torch.poisson(torch.full((2, 20, 1, 700), 2.0))
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
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    assert grads and all(torch.isfinite(g).all() for g in grads)
                    assert any(torch.count_nonzero(g) for g in grads)
                    optimizer.step()
                    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
                    print(f"PASS {device} {architecture} {sg} elife={elife} {type(model).__name__}", flush=True)
                    del model, optimizer, before, grads, output, loss
                    gc.collect()
    print("PASS original model forward/backward, state reset and optimizer updates")


if __name__ == "__main__":
    main()
