from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .dataset import DIV2KRestorationDataset
from .model import build_model


def _read_scalars(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    event_dir = run_dir / "tensorboard"
    if not event_dir.exists():
        raise FileNotFoundError(f"Missing TensorBoard directory: {event_dir}")

    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()
    scalars: dict[str, tuple[list[int], list[float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        scalars[tag] = ([event.step for event in events], [event.value for event in events])
    return scalars


def _read_history(run_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    history_path = run_dir / "metrics_history.csv"
    if not history_path.exists():
        return {}

    curves: dict[str, tuple[list[int], list[float]]] = {
        "loss": ([], []),
        "psnr": ([], []),
        "ssim": ([], []),
    }
    with history_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            curves["loss"][0].append(epoch)
            curves["loss"][1].append(float(row["loss"]))
            if row.get("psnr"):
                curves["psnr"][0].append(epoch)
                curves["psnr"][1].append(float(row["psnr"]))
            if row.get("ssim"):
                curves["ssim"][0].append(epoch)
                curves["ssim"][1].append(float(row["ssim"]))
    return curves


def _plot_line(
    points: tuple[list[int], list[float]] | None,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not points or not points[0]:
        print(f"Skip {title}: no data found.")
        return

    steps, values = points
    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, marker="o", linewidth=1.8)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(steps)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metric_curves(run_dir: Path, output_dir: Path) -> None:
    history = _read_history(run_dir)
    if history:
        curves = history
    else:
        scalars = _read_scalars(run_dir)
        curves = {
            "loss": scalars.get("epoch/loss"),
            "psnr": scalars.get("valid/psnr"),
            "ssim": scalars.get("valid/ssim"),
        }
    _plot_line(curves.get("loss"), "Epoch Training Loss", "Loss", output_dir / "loss_curve.png")
    _plot_line(curves.get("psnr"), "Epoch Validation PSNR", "PSNR", output_dir / "psnr_curve.png")
    _plot_line(curves.get("ssim"), "Epoch Validation SSIM", "SSIM", output_dir / "ssim_curve.png")


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(tensor.detach().cpu().clamp(0, 1))


def _load_trained_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint.get("variant", "prompt_lora"),
        dim=checkpoint.get("dim", 24),
        lora_rank=checkpoint.get("lora_rank", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def _build_untrained_like(checkpoint: dict, device: torch.device):
    model = build_model(
        checkpoint.get("variant", "prompt_lora"),
        dim=checkpoint.get("dim", 24),
        lora_rank=checkpoint.get("lora_rank", 4),
    ).to(device)
    model.eval()
    return model


def _draw_comparison(images: list[Image.Image], titles: list[str], output_path: Path) -> None:
    plt.figure(figsize=(16, 5))
    for idx, (image, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(1, len(images), idx)
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


@torch.no_grad()
def save_comparisons(
    run_dir: Path,
    output_dir: Path,
    data_root: str,
    degradation: str,
    num_samples: int,
    device: torch.device,
) -> None:
    checkpoint_path = run_dir / "best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "last.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best.pth or last.pth found in {run_dir}")

    trained_model, checkpoint = _load_trained_model(checkpoint_path, device)
    untrained_model = _build_untrained_like(checkpoint, device)
    dataset = DIV2KRestorationDataset(data_root, split="valid", degradations=[degradation], limit=num_samples)

    for index in range(len(dataset)):
        sample = dataset[index]
        lq = sample["lq"].unsqueeze(0).to(device)
        gt = sample["gt"]
        degradation_id = sample["degradation"].unsqueeze(0).to(device)
        mix_weights = sample["mix_weights"].unsqueeze(0).to(device)

        untrained = untrained_model(lq, degradation_id, mix_weights).squeeze(0)
        trained = trained_model(lq, degradation_id, mix_weights).squeeze(0)
        lq_cpu = sample["lq"]

        images = [
            _tensor_to_image(gt),
            _tensor_to_image(lq_cpu),
            _tensor_to_image(untrained),
            _tensor_to_image(trained),
        ]
        titles = ["GT Clean", "LQ Degraded", "Untrained Model", "Trained Model"]
        name = f"{index + 1:02d}_{sample['degradation_name']}_{sample['name']}.png"
        _draw_comparison(images, titles, output_dir / "comparisons" / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training curves and restoration comparisons.")
    parser.add_argument("--run-dir", required=True, help="A run directory containing tensorboard/ and best.pth or last.pth.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--degradation", default="mixed", choices=["lowlight", "blur", "jpeg", "haze", "mixed"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    device = torch.device(args.device)

    save_metric_curves(run_dir, output_dir)
    save_comparisons(run_dir, output_dir, args.data_root, args.degradation, args.num_samples, device)
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
