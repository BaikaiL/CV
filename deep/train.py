from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--degradations", nargs="*", default=list(ALL_DEGRADATIONS), choices=list(ALL_DEGRADATIONS))
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only LoRA, prompts, and output/refine heads.")
    parser.add_argument("--output-dir", default="results/deep")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    train_set = DIV2KRestorationDataset(args.data_root, split="train", degradations=args.degradations, limit=args.train_limit)
    valid_set = DIV2KRestorationDataset(args.data_root, split="valid", degradations=args.degradations, limit=args.valid_limit)
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

    best_psnr = -1.0
    global_step = 0
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

        valid_metrics = validate(model, valid_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("epoch/loss", avg_loss, epoch)
        writer.add_scalar("valid/psnr", valid_metrics["psnr"], epoch)
        writer.add_scalar("valid/ssim", valid_metrics["ssim"], epoch)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, PSNR={valid_metrics['psnr']:.3f}, SSIM={valid_metrics['ssim']:.4f}")

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
        if valid_metrics["psnr"] > best_psnr:
            best_psnr = valid_metrics["psnr"]
            torch.save(checkpoint, output_dir / "best.pth")
    writer.close()


if __name__ == "__main__":
    main()
