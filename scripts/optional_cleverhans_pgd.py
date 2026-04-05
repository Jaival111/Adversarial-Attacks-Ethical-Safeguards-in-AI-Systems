"""
Optional Phase 2 cross-check: verify CleverHans PyTorch PGD is importable alongside this repo.

Install extras: pip install cleverhans

The main training stack uses native PyTorch PGD (`adversarial_safeguards.attacks.pgd_attack`) and
Torchattacks for parity checks — both follow the standard l_inf PGD recipe used in CleverHans tutorials.
"""
from __future__ import annotations


def main() -> None:
    try:
        from cleverhans.torch.attacks.projected_gradient_descent import (  # type: ignore
            projected_gradient_descent,
        )

        print("CleverHans Torch PGD available:", projected_gradient_descent)
    except Exception as e:
        print("CleverHans not available:", e)
        print("Install with: pip install cleverhans")


if __name__ == "__main__":
    main()
