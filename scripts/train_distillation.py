"""
Phase 3: Defensive distillation — train a student against a high-temperature teacher (baseline bundle).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from adversarial_safeguards.bundle import load_serving_bundle, save_serving_bundle
from adversarial_safeguards.data.cifar import get_cifar10_loader
from adversarial_safeguards.defenses.distillation import distillation_loss
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
    p.add_argument("--teacher-bundle", type=str, default="./artifacts/baseline_bundle.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out", type=str, default="./artifacts/distill_bundle.pt")
    p.add_argument("--detector-batches", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    teacher, _, _ = load_serving_bundle(args.teacher_bundle, device)
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    train_loader = get_cifar10_loader(
        args.batch_size, data_dir=args.data_dir, train=True, num_workers=0, pin_memory=pin
    )
    test_loader = get_cifar10_loader(
        args.batch_size, data_dir=args.data_dir, train=False, num_workers=0, pin_memory=pin
    )

    student = CifarCNN().to(device)
    opt = optim.AdamW(student.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        student.train()
        bar = tqdm(train_loader, desc=f"distill {epoch+1}/{args.epochs}")
        for x, y in bar:
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            with torch.no_grad():
                t_logits = teacher(x)
            opt.zero_grad(set_to_none=True)
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y, temperature=args.temperature, alpha=args.alpha)
            loss.backward()
            opt.step()
            bar.set_postfix(loss=float(loss.item()))
        sched.step()
        acc = accuracy(student, test_loader, device)
        print(f"epoch {epoch+1} student test acc: {acc:.4f}")

    final_acc = accuracy(student, test_loader, device)
    det_state = collect_detector_state(student, test_loader, device, max_batches=args.detector_batches)
    save_serving_bundle(
        args.out,
        student.cpu(),
        det_state,
        meta={
            "kind": "defensive_distillation",
            "model_version": "1.0.0",
            "teacher": str(Path(args.teacher_bundle).resolve()),
            "temperature": args.temperature,
            "kd_alpha": args.alpha,
            "test_acc": final_acc,
        },
    )
    print(f"Saved {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
