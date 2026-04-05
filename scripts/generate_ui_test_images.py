"""
Generate random 32×32 RGB PNGs for exercising the web UI / predict API.

Usage:
  python scripts/generate_ui_test_images.py
  python scripts/generate_ui_test_images.py --out ./artifacts/ui_test_images --count 12 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 32


def noise(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def solid(rng: np.random.Generator) -> np.ndarray:
    c = rng.integers(30, 226, size=3, dtype=np.uint8)
    return np.tile(c, (SIZE, SIZE, 1))


def gradient(rng: np.random.Generator) -> np.ndarray:
    c0 = rng.integers(0, 200, size=3, dtype=np.uint8)
    c1 = rng.integers(55, 256, size=3, dtype=np.uint8)
    gx = np.linspace(0, 1, SIZE, dtype=np.float32)
    gy = np.linspace(0, 1, SIZE, dtype=np.float32)
    u = (gx[None, :, None] + gy[:, None, None]) / 2.0
    arr = c0.astype(np.float32) * (1 - u) + c1.astype(np.float32) * u
    return np.clip(arr, 0, 255).astype(np.uint8)


def stripes_highfreq(rng: np.random.Generator) -> np.ndarray:
    """Sharp vertical stripes (stress-test JPEG defense / detector)."""
    base = rng.integers(40, 220, size=3, dtype=np.uint8)
    alt = rng.integers(0, 256, size=3, dtype=np.uint8)
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for x in range(SIZE):
        img[:, x] = base if x % 2 == 0 else alt
    return img


def soft_blobs(rng: np.random.Generator) -> np.ndarray:
    """Smooth color blobs (more natural-ish at 32×32)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy = rng.uniform(4, SIZE - 4, size=2)
    r = rng.uniform(6, 14)
    mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2))
    c0 = rng.integers(0, 256, size=3).astype(np.float32)
    c1 = rng.integers(0, 256, size=3).astype(np.float32)
    arr = c0 * (1 - mask[..., None]) + c1 * mask[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8)


def pil_shapes(rng: np.random.Generator) -> np.ndarray:
    """Simple shapes on random background (PIL)."""
    bg = tuple(int(x) for x in rng.integers(40, 220, size=3))
    im = Image.new("RGB", (SIZE, SIZE), bg)
    dr = ImageDraw.Draw(im)
    for _ in range(rng.integers(2, 6)):
        a = (int(rng.integers(0, SIZE // 2)), int(rng.integers(0, SIZE // 2)))
        b = (int(rng.integers(SIZE // 2, SIZE)), int(rng.integers(SIZE // 2, SIZE)))
        fill = tuple(int(x) for x in rng.integers(0, 256, size=3))
        if rng.random() < 0.5:
            dr.ellipse([a[0], a[1], b[0], b[1]], fill=fill, outline=None)
        else:
            dr.rectangle([a[0], a[1], b[0], b[1]], fill=fill, outline=None)
    return np.array(im)


GENERATORS = (
    ("noise", noise),
    ("solid", solid),
    ("gradient", gradient),
    ("stripes_hf", stripes_highfreq),
    ("blobs", soft_blobs),
    ("shapes", pil_shapes),
)


def main() -> None:
    p = argparse.ArgumentParser(description="Write PNG test images for the UI/API.")
    p.add_argument("--out", type=str, default="./artifacts/ui_test_images", help="Output directory")
    p.add_argument("--count", type=int, default=10, help="How many images to write")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.count):
        name, fn = GENERATORS[i % len(GENERATORS)]
        arr = fn(rng)
        path = out / f"sample_{i:02d}_{name}.png"
        Image.fromarray(arr, mode="RGB").save(path)

    print(f"Wrote {args.count} PNG files to {out.resolve()}")
    print("Upload any of them in the React UI (or POST to /v1/predict).")


if __name__ == "__main__":
    main()
