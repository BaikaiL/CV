from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm

from .dataset import DEGRADATION_TO_INDEX, _mix_weights
from .model import build_model


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    tensor = pil_to_tensor(Image.open(path).convert("RGB")).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = to_pil_image(tensor.squeeze(0).detach().cpu().clamp(0, 1))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run restoration inference with a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output image file or directory.")
    parser.add_argument("--degradation", default="mixed", choices=list(DEGRADATION_TO_INDEX))
    parser.add_argument("--mixed-params", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint.get("variant", "prompt_lora"),
        dim=checkpoint.get("dim", 32),
        lora_rank=checkpoint.get("lora_rank", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    input_path = Path(args.input)
    output_path = Path(args.output)
    paths = sorted(input_path.glob("*.png")) if input_path.is_dir() else [input_path]
    degradation = torch.tensor([DEGRADATION_TO_INDEX[args.degradation]], dtype=torch.long, device=device)
    mix_weights = _mix_weights(args.degradation, args.mixed_params).unsqueeze(0).to(device)

    with torch.no_grad():
        for path in tqdm(paths, desc="infer"):
            restored = model(_load_image(path, device), degradation, mix_weights)
            target = output_path / path.name if input_path.is_dir() else output_path
            _save_image(restored, target)


if __name__ == "__main__":
    main()

