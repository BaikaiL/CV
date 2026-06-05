"""Generate synthetic DIV2K degradations for restoration experiments.

Example:
    python degradations/generate_dataset.py --data-root data --patches-per-train-image 8
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from degradations.haze import add_haze
    from degradations.jpeg import add_jpeg_artifacts
    from degradations.lowlight import add_lowlight
    from degradations.motion_blur import add_motion_blur
except ModuleNotFoundError:
    from haze import add_haze
    from jpeg import add_jpeg_artifacts
    from lowlight import add_lowlight
    from motion_blur import add_motion_blur


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEGRADATIONS = ("lowlight", "blur", "jpeg", "haze", "mixed")


@dataclass(frozen=True)
class PatchConfig:
    patch_size: int
    patches_per_image: int
    center_crop: bool = False


def read_rgb(path: Path) -> np.ndarray:
    try:
        image = Image.open(path).convert("RGB")
    except OSError as exc:
        raise RuntimeError(f"Failed to read image: {path}")
    return np.asarray(image, dtype=np.uint8)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def list_images(root: Path) -> list[Path]:
    images: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError:
            print(f"[warn] skip unreadable image: {path}")
            continue
        images.setdefault(path.stem, path)
    return sorted(images.values())


def crop_patches(image: np.ndarray, config: PatchConfig, rng: np.random.Generator) -> list[np.ndarray]:
    height, width = image.shape[:2]
    patch_size = min(config.patch_size, height, width)

    if config.center_crop:
        top = max((height - patch_size) // 2, 0)
        left = max((width - patch_size) // 2, 0)
        return [image[top : top + patch_size, left : left + patch_size].copy()]

    patches: list[np.ndarray] = []
    max_top = max(height - patch_size, 0)
    max_left = max(width - patch_size, 0)
    for _ in range(config.patches_per_image):
        top = int(rng.integers(0, max_top + 1)) if max_top else 0
        left = int(rng.integers(0, max_left + 1)) if max_left else 0
        patches.append(image[top : top + patch_size, left : left + patch_size].copy())
    return patches


def degrade_once(image: np.ndarray, degradation: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    if degradation == "lowlight":
        gamma = float(rng.uniform(1.7, 2.8))
        gain = float(rng.uniform(0.42, 0.72))
        noise_std = float(rng.uniform(0.006, 0.020))
        return add_lowlight(image, gamma=gamma, gain=gain, noise_std=noise_std, rng=rng), (
            f"gamma={gamma:.3f};gain={gain:.3f};noise_std={noise_std:.4f}"
        )

    if degradation == "blur":
        length = int(rng.integers(11, 32))
        if length % 2 == 0:
            length += 1
        angle = float(rng.uniform(0.0, 180.0))
        noise_std = float(rng.uniform(0.001, 0.007))
        return add_motion_blur(image, length=length, angle=angle, noise_std=noise_std, rng=rng), (
            f"length={length};angle={angle:.2f};noise_std={noise_std:.4f}"
        )

    if degradation == "jpeg":
        quality = int(rng.integers(12, 46))
        return add_jpeg_artifacts(image, quality=quality), f"quality={quality}"

    if degradation == "haze":
        beta = float(rng.uniform(1.0, 2.6))
        air = tuple(float(v) for v in rng.uniform(0.82, 0.96, size=3))
        return add_haze(image, beta=beta, airlight=air, rng=rng), (
            f"beta={beta:.3f};airlight=({air[0]:.3f},{air[1]:.3f},{air[2]:.3f})"
        )

    raise ValueError(f"Unknown degradation: {degradation}")


def degrade_mixed(image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    count = int(rng.integers(2, 4))
    selected = list(rng.choice(("lowlight", "blur", "jpeg", "haze"), size=count, replace=False))
    degraded = image.copy()
    details: list[str] = []
    for name in selected:
        degraded, params = degrade_once(degraded, name, rng)
        details.append(f"{name}[{params}]")
    return degraded, "+".join(selected) + "|" + "|".join(details)


def clear_generated_dirs(data_root: Path, splits: tuple[str, ...]) -> None:
    for split in splits:
        for folder in (data_root / f"{split}_GT", data_root / f"{split}_LQ"):
            if folder.exists():
                shutil.rmtree(folder)


def generate_split(
    split: str,
    hr_dir: Path,
    data_root: Path,
    config: PatchConfig,
    rng: np.random.Generator,
    metadata_rows: list[dict[str, str]],
    limit: int | None = None,
) -> None:
    images = list_images(hr_dir)
    if limit is not None:
        images = images[:limit]

    gt_root = data_root / f"{split}_GT"
    lq_root = data_root / f"{split}_LQ"

    for image_index, image_path in enumerate(images, start=1):
        clean = read_rgb(image_path)
        patches = crop_patches(clean, config, rng)
        for patch_index, patch in enumerate(patches, start=1):
            stem = f"{image_path.stem}_p{patch_index:03d}"
            gt_path = gt_root / f"{stem}.png"
            write_rgb(gt_path, patch)

            for degradation in DEGRADATIONS:
                if degradation == "mixed":
                    degraded, params = degrade_mixed(patch, rng)
                else:
                    degraded, params = degrade_once(patch, degradation, rng)

                lq_path = lq_root / degradation / f"{stem}.png"
                write_rgb(lq_path, degraded)
                metadata_rows.append(
                    {
                        "split": split,
                        "source": str(image_path.as_posix()),
                        "gt": str(gt_path.as_posix()),
                        "lq": str(lq_path.as_posix()),
                        "degradation": degradation,
                        "params": params,
                    }
                )

        print(f"[{split}] {image_index:04d}/{len(images):04d} {image_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DIV2K synthetic restoration degradations.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-hr", type=Path, default=None)
    parser.add_argument("--valid-hr", type=Path, default=None)
    parser.add_argument("--train-patch-size", type=int, default=256)
    parser.add_argument("--valid-patch-size", type=int, default=512)
    parser.add_argument("--patches-per-train-image", type=int, default=8)
    parser.add_argument("--patches-per-valid-image", type=int, default=1)
    parser.add_argument("--valid-center-crop", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    train_hr = args.train_hr or data_root / "DIV2K_train_HR"
    valid_hr = args.valid_hr or data_root / "DIV2K_valid_HR"

    if not train_hr.exists():
        raise FileNotFoundError(f"Training HR directory not found: {train_hr}")
    if not valid_hr.exists():
        raise FileNotFoundError(f"Validation HR directory not found: {valid_hr}")

    if args.overwrite:
        clear_generated_dirs(data_root, ("train", "valid"))

    rng = np.random.default_rng(args.seed)
    metadata_rows: list[dict[str, str]] = []
    generate_split(
        split="train",
        hr_dir=train_hr,
        data_root=data_root,
        config=PatchConfig(args.train_patch_size, args.patches_per_train_image),
        rng=rng,
        metadata_rows=metadata_rows,
        limit=args.limit_train,
    )
    generate_split(
        split="valid",
        hr_dir=valid_hr,
        data_root=data_root,
        config=PatchConfig(args.valid_patch_size, args.patches_per_valid_image, args.valid_center_crop),
        rng=rng,
        metadata_rows=metadata_rows,
        limit=args.limit_valid,
    )

    metadata_path = data_root / "degradation_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "gt", "lq", "degradation", "params"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Done. Wrote {len(metadata_rows)} LQ images and metadata to {metadata_path}")


if __name__ == "__main__":
    main()
