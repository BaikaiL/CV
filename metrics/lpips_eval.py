from __future__ import annotations

import argparse
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm


def _load(path: Path, device: torch.device) -> torch.Tensor:
    tensor = pil_to_tensor(Image.open(path).convert("RGB")).float().unsqueeze(0) / 255.0
    return tensor.to(device) * 2.0 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LPIPS between restored images and GT images.")
    parser.add_argument("--restored-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    restored_dir = Path(args.restored_dir)
    gt_dir = Path(args.gt_dir)
    scores: list[float] = []
    with torch.no_grad():
        for restored in tqdm(sorted(restored_dir.glob("*.png"))):
            gt = gt_dir / restored.name
            if not gt.exists():
                continue
            score = metric(_load(restored, device), _load(gt, device)).item()
            scores.append(score)
    if not scores:
        raise RuntimeError("No matched image pairs found.")
    print(f"LPIPS: {sum(scores) / len(scores):.6f} ({len(scores)} images)")


if __name__ == "__main__":
    main()

