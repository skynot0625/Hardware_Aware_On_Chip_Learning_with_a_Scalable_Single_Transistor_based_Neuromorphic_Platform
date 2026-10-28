"""Plain PyTorch training, with argparse to keep the review example dependency-light."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data import load_data
from models import PreActResNet18, SHDELife


def loss_for(scores: Tensor, labels: Tensor, dataset: str) -> Tensor:
    target = F.one_hot(labels, scores.shape[1]).to(scores.dtype)
    if dataset == "shd":
        return F.mse_loss(scores, target)
    # Active loss in the original SpikingMNISTLitModule.model_step.
    return (0.95 * F.cross_entropy(scores, labels) + 0.05 * F.mse_loss(scores, target)) / 5


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None,
              dataset: str) -> tuple[float, float]:
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    total_loss = correct = count = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            scores = model(images)
            loss = loss_for(scores, labels, dataset)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        correct += int((scores.argmax(1) == labels).sum())
        count += labels.numel()
    return total_loss / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "shd"], default="cifar10")
    parser.add_argument("--surrogate", choices=["sg", "csg"], default="csg")
    parser.add_argument("--elife", action="store_true", help="enable backward eligibility in either network")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--epochs", type=int, default=None, help="default: CIFAR 300, SHD 100")
    parser.add_argument("--scale", type=int, default=None, help="default: CIFAR 64, SHD 1024")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default: CIFAR 1e-4, SHD 1e-3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="limit each split for a smoke run only")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (100 if args.dataset == "shd" else 300)
    args.scale = args.scale if args.scale is not None else (1024 if args.dataset == "shd" else 64)
    args.lr = args.lr if args.lr is not None else (1e-3 if args.dataset == "shd" else 1e-4)
    if min(args.epochs, args.scale, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, scale, batch-size, threads and lr must be positive")
    if args.workers < 0 or (args.limit is not None and args.limit < 1):
        parser.error("workers must be nonnegative and limit must be positive")
    if args.dataset == "shd" and args.scale != 1024:
        parser.error("the original SHD implementation requires --scale 1024")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.output.exists():
        parser.error("output directory exists; choose a new --output to preserve prior runs")
    if args.dataset == "shd" and not all((args.data_dir / name).is_file() for name in ["shd_train.h5", "shd_test.h5"]):
        parser.error("SHD requires shd_train.h5 and shd_test.h5 in --data-dir; see README")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    splits = load_data(args.dataset, args.data_dir, args.seed)
    if args.limit is not None:
        splits = tuple(Subset(split, range(min(args.limit, len(split)))) for split in splits)
    loaders = [DataLoader(split, batch_size=args.batch_size, shuffle=index == 0,
                          num_workers=args.workers) for index, split in enumerate(splits)]
    surrogate = args.surrogate
    model = (SHDELife(scale=args.scale, surrogate=surrogate, elife=args.elife) if args.dataset == "shd" else
             PreActResNet18(scale=args.scale, elife=args.elife, surrogate=surrogate)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best = -1.0
    with (args.output / "metrics.jsonl").open("w") as log:
        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(model, loaders[0], optimizer, args.dataset)
            val_loss, val_acc = run_epoch(model, loaders[1], None, args.dataset)
            record = dict(epoch=epoch + 1, train_loss=train_loss, train_accuracy=train_acc,
                          val_loss=val_loss, val_accuracy=val_acc, lr=scheduler.get_last_lr()[0])
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if val_acc > best:
                best = val_acc
                torch.save(model.state_dict(), args.output / "best.pt")
            scheduler.step()
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=args.device, weights_only=True))
    test_loss, test_acc = run_epoch(model, loaders[2], None, args.dataset)
    result = dict(test_loss=test_loss, test_accuracy=test_acc, smoke_run=args.limit is not None)
    (args.output / "test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
