# Adversarial attacks, proactive defenses, and ethical safeguards

End-to-end research scaffold for **CIFAR-10**: baseline training, **PGD / FGSM** attacks (native PyTorch + **Torchattacks** parity), **adversarial training**, **input transformations** (JPEG + smoothing), **defensive distillation**, **statistical detection** (KL to a clean prior, confidence instability), **Grad-CAM**, structured **logging**, an **ethical risk tier** (high / medium / low), and a **transparency JSON** report. Deployment: **FastAPI** + **React (Vite)**.

## Architecture (pipeline)

```text
Input → Adversarial detector → Robust model (optional input defense) → Monitoring & logs → Risk / transparency API
```

## Repository layout

| Path | Role |
|------|------|
| `adversarial_safeguards/` | Library: models, attacks, defenses, detection, monitoring, risk framework, `RobustInferencePipeline` |
| `scripts/train_baseline.py` | Phase 1 — baseline model + detector calibration bundle |
| `scripts/run_attack_eval.py` | Phase 2 — clean vs FGSM / PGD / Torchattacks PGD accuracy |
| `scripts/train_adversarial.py` | Phase 3 — adversarial training bundle |
| `scripts/train_distillation.py` | Phase 3 — defensive distillation student bundle |
| `scripts/optional_cleverhans_pgd.py` | Optional CleverHans import check (see note below) |
| `scripts/generate_ui_test_images.py` | Random 32×32 PNGs for UI/API smoke tests → `artifacts/ui_test_images/` |
| `api/main.py` | Phase 7 — FastAPI: `/health`, `/v1/predict`, `/v1/transparency/{id}` |
| `frontend/` | Phase 7 — React UI posting images to the API (via Vite proxy) |

## Setup (Python)

```powershell
cd Adversarial-Attacks-Ethical-Safeguards-in-AI-Systems
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# If torch fails to resolve on Windows, install CPU wheels first:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Train and evaluate

```powershell
# Phase 1 — produces artifacts/baseline_bundle.pt (weights + detector stats)
python scripts/train_baseline.py --epochs 15 --out artifacts/baseline_bundle.pt

# Phase 2 — attack accuracy drop (uses bundle from Phase 1)
python scripts/run_attack_eval.py --bundle artifacts/baseline_bundle.pt

# Phase 3 — defenses
python scripts/train_adversarial.py --out artifacts/adv_train_bundle.pt
python scripts/train_distillation.py --teacher-bundle artifacts/baseline_bundle.pt --out artifacts/distill_bundle.pt
```

Serve with a trained bundle:

```powershell
$env:SERVING_BUNDLE="artifacts/baseline_bundle.pt"
.\.venv\Scripts\python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`. Open the printed URL (default `http://localhost:5173`), upload an image, and review the structured transparency report (raw JSON is available behind “Raw JSON”).

Generate sample uploads:

```powershell
python scripts/generate_ui_test_images.py --count 10 --out artifacts/ui_test_images
```

## CleverHans note

The **reference PGD recipe** (l∞, random start, sign gradients) matches CleverHans / Madry-style PGD. This repo implements PGD in PyTorch and uses **Torchattacks** for a second PGD implementation in `run_attack_eval.py`. For CleverHans specifically:

```powershell
pip install cleverhans
python scripts/optional_cleverhans_pgd.py
```

Optional TensorFlow stack: `requirements-cleverhans.txt`.

## Ethics and limitations

Detectors and risk tiers are **heuristic** (not certified robustness). Transparency reports are meant for **governance, logging, and accountable disclosure** of model behavior under stress — not as a guarantee against adaptive attackers.

## License

Use and cite responsibly; adversarial ML code is for **defensive research and education** only.
