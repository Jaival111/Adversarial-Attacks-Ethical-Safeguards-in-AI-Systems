"""
Phase 3: Adversarial training (PGD inner loop) + export serving bundle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from adversarial_safeguards.bundle import save_serving_bundle
from adversarial_safeguards.data.cifar import get_cifar10_loader
from adversarial_safeguards.defenses.adversarial_training import adversarial_loss_batch
from adversarial_safeguards.detection.statistical import collect_detector_state
from adversarial_safeguards.models.cifar_cnn import CifarCNN


def accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="./artifacts/adv_train_bundle.pt")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--detector-batches", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_loader = get_cifar10_loader(
        args.batch_size, data_dir=args.data_dir, train=True, num_workers=0, pin_memory=pin
    )
    test_loader = get_cifar10_loader(
        args.batch_size, data_dir=args.data_dir, train=False, num_workers=0, pin_memory=pin
    )

    model = CifarCNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(train_loader, desc=f"adv epoch {epoch+1}/{args.epochs}")
        for x, y in bar:
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            loss = adversarial_loss_batch(
                model, x, y, eps=args.eps, alpha=args.alpha, pgd_steps=args.pgd_steps, clean_weight=0.5
            )
            loss.backward()
            opt.step()
            bar.set_postfix(loss=float(loss.item()))
        sched.step()
        acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch+1} clean test acc: {acc:.4f}")

    final_acc = accuracy(model, test_loader, device)
    det_state = collect_detector_state(model, test_loader, device, max_batches=args.detector_batches)
    save_serving_bundle(
        args.out,
        model.cpu(),
        det_state,
        meta={
            "kind": "adversarial_training",
            "model_version": "1.0.0",
            "epochs": args.epochs,
            "eps": args.eps,
            "pgd_steps": args.pgd_steps,
            "test_acc_clean": final_acc,
        },
    )
    print(f"Saved {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
