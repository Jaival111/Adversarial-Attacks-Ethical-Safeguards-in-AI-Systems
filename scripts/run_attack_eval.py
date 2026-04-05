"""
Phase 2: Evaluate clean vs adversarial accuracy (PGD / FGSM; Torchattacks PGD for cross-check).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from adversarial_safeguards.attacks.pgd_fgsm import fgsm_attack, pgd_attack, pgd_with_torchattacks
from adversarial_safeguards.bundle import load_serving_bundle
from adversarial_safeguards.data.cifar import get_cifar10_loader


@torch.no_grad()
def eval_clean(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="clean"):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def eval_attack(
    model: nn.Module,
    loader,
    device: torch.device,
    attack: str,
    eps: float,
    pgd_steps: int,
    alpha: float,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc=attack):
        x, y = x.to(device), y.to(device)
        if attack == "pgd":
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=pgd_steps, random_start=True)
        elif attack == "fgsm":
            x_adv = fgsm_attack(model, x, y, eps=eps)
        elif attack == "pgd_ta":
            x_adv = pgd_with_torchattacks(model, x, y, eps=eps, alpha=alpha, steps=pgd_steps)
        else:
            raise ValueError(attack)
        pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=str, default="./artifacts/baseline_bundle.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--limit-batches", type=int, default=0, help="0 = full test set")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()
    model, _, meta = load_serving_bundle(args.bundle, device)

    test_loader = get_cifar10_loader(
        args.batch_size, data_dir=args.data_dir, train=False, num_workers=0, pin_memory=pin
    )
    if args.limit_batches > 0:

        class SubsetLoader:
            def __init__(self, loader, n: int):
                self.loader = loader
                self.n = n

            def __iter__(self):
                for i, batch in enumerate(self.loader):
                    if i >= self.n:
                        break
                    yield batch

        test_loader = SubsetLoader(test_loader, args.limit_batches)  # type: ignore[assignment]

    clean = eval_clean(model, test_loader, device)
    pgd = eval_attack(model, test_loader, device, "pgd", args.eps, args.pgd_steps, args.alpha)
    fgsm = eval_attack(model, test_loader, device, "fgsm", args.eps, args.pgd_steps, args.alpha)
    pgd_ta = eval_attack(model, test_loader, device, "pgd_ta", args.eps, args.pgd_steps, args.alpha)

    print("--- Attack evaluation (lower is worse under attack) ---")
    print(f"Bundle meta: {meta}")
    print(f"Clean accuracy:     {clean:.4f}")
    print(f"Under PGD (ours):   {pgd:.4f}  (drop {clean - pgd:.4f})")
    print(f"Under FGSM:         {fgsm:.4f}  (drop {clean - fgsm:.4f})")
    print(f"Under PGD (TA):     {pgd_ta:.4f}  (drop {clean - pgd_ta:.4f})")


if __name__ == "__main__":
    main()
