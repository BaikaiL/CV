from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics.psnr_ssim import calculate_psnr, calculate_ssim

from .dataset import ALL_DEGRADATIONS, DIV2KRestorationDataset
from .lora import mark_only_lora_prompt_trainable
from .losses import RestorationLoss
from .model import build_model


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _forward(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    return model(batch["lq"], batch["degradation"], batch["mix_weights"])


def _limit_dataset(dataset, limit: int | None, seed: int) -> Subset | object:
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _write_history(output_dir: Path, history: list[dict[str, float | int | bool]]) -> None:
    if not history:
        return
    csv_path = output_dir / "metrics_history.csv"
    fieldnames = ["epoch", "loss", "psnr", "ssim", "validated", "best_psnr"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for batch in tqdm(loader, desc="valid", leave=False):
        batch = _move_batch(batch, device)
        output = _forward(model, batch)
        psnr_values.extend(calculate_psnr(output, batch["gt"]).detach().cpu().tolist())
        ssim_values.extend(calculate_ssim(output, batch["gt"]).detach().cpu().tolist())
    return {
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prompt-LoRA Restormer on generated DIV2K degradations.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--variant", default="prompt_lora", choices=["baseline", "prompt", "lora", "prompt_lora"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--course-train-samples", type=int, default=1200, help="Samples used per epoch in course mode.")
    parser.add_argument("--course-valid-samples", type=int, default=40, help="Validation samples used in course mode.")
    parser.add_argument("--val-interval", type=int, default=2, help="Run validation every N epochs.")
    parser.add_argument("--early-stop-patience", type=int, default=4, help="Stop if PSNR does not improve for this many validations.")
    parser.add_argument("--min-psnr-gain", type=float, default=0.01, help="Minimum PSNR gain counted as an improvement.")
    parser.add_argument("--full-training", action="store_true", help="Use full datasets and validate every epoch.")
    parser.add_argument("--degradations", nargs="*", default=list(ALL_DEGRADATIONS), choices=list(ALL_DEGRADATIONS))
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only LoRA, prompts, and output/refine heads.")
    parser.add_argument("--output-dir", default="results/deep")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.manual_seed(args.seed)

    train_limit = args.train_limit
    valid_limit = args.valid_limit
    if not args.full_training:
        train_limit = train_limit or args.course_train_samples
        valid_limit = valid_limit or args.course_valid_samples

    train_set = DIV2KRestorationDataset(args.data_root, split="train", degradations=args.degradations)
    valid_set = DIV2KRestorationDataset(args.data_root, split="valid", degradations=args.degradations)
    train_set = _limit_dataset(train_set, train_limit, args.seed)
    valid_set = _limit_dataset(valid_set, valid_limit, args.seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args.variant, dim=args.dim, lora_rank=args.lora_rank).to(device)
    if args.freeze_backbone:
        mark_only_lora_prompt_trainable(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    criterion = RestorationLoss().to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    run_name = f"{args.variant}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "tensorboard")
    writer.add_text("config/params", str(vars(args)))
    writer.add_scalar("params/total", total, 0)
    writer.add_scalar("params/trainable", trainable, 0)
    print(f"Device: {device} | trainable params: {trainable:,}/{total:,}")
    print(
        "Training plan: "
        f"train_samples={len(train_set)}, valid_samples={len(valid_set)}, "
        f"epochs={args.epochs}, val_interval={1 if args.full_training else args.val_interval}"
    )

    best_psnr = -1.0
    best_metrics = {"psnr": 0.0, "ssim": 0.0}
    stale_validations = 0
    global_step = 0
    history: list[dict[str, float | int | bool]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                output = _forward(model, batch)
                loss, parts = criterion(output, batch["gt"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), global_step)
            for key, value in parts.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        should_validate = args.full_training or epoch == args.epochs or epoch % args.val_interval == 0
        if should_validate:
            valid_metrics = validate(model, valid_loader, device)
            writer.add_scalar("valid/psnr", valid_metrics["psnr"], epoch)
            writer.add_scalar("valid/ssim", valid_metrics["ssim"], epoch)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, PSNR={valid_metrics['psnr']:.3f}, SSIM={valid_metrics['ssim']:.4f}")
        else:
            valid_metrics = best_metrics
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, validation skipped")

        checkpoint = {
            "epoch": epoch,
            "variant": args.variant,
            "dim": args.dim,
            "lora_rank": args.lora_rank,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": valid_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        improved = should_validate and valid_metrics["psnr"] > best_psnr + args.min_psnr_gain
        if improved:
            best_psnr = valid_metrics["psnr"]
            best_metrics = valid_metrics
            stale_validations = 0
            torch.save(checkpoint, output_dir / "best.pth")
        should_stop = False
        if should_validate and not improved:
            stale_validations += 1
            if stale_validations >= args.early_stop_patience:
                print(f"Early stop: PSNR did not improve for {stale_validations} validations.")
                should_stop = True
        history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "psnr": valid_metrics["psnr"] if should_validate else "",
                "ssim": valid_metrics["ssim"] if should_validate else "",
                "validated": should_validate,
                "best_psnr": best_psnr,
            }
        )
        _write_history(output_dir, history)
        if should_stop:
            break
    writer.close()


if __name__ == "__main__":
    main()
