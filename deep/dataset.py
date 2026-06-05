from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


BASE_DEGRADATIONS = ("lowlight", "blur", "jpeg", "haze")
ALL_DEGRADATIONS = BASE_DEGRADATIONS + ("mixed",)
DEGRADATION_TO_INDEX = {name: idx for idx, name in enumerate(ALL_DEGRADATIONS)}


@dataclass(frozen=True)
class RestorationSample:
    gt: Path
    lq: Path
    degradation: str
    params: str


def _read_metadata(data_root: Path, split: str) -> list[RestorationSample]:
    metadata_path = data_root / "degradation_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    samples: list[RestorationSample] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            samples.append(
                RestorationSample(
                    gt=data_root.parent / row["gt"],
                    lq=data_root.parent / row["lq"],
                    degradation=row["degradation"],
                    params=row.get("params", ""),
                )
            )
    if not samples:
        raise RuntimeError(f"No samples found for split={split!r} in {metadata_path}")
    return samples


def _image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _mix_weights(degradation: str, params: str) -> torch.Tensor:
    weights = torch.zeros(len(BASE_DEGRADATIONS), dtype=torch.float32)
    if degradation != "mixed":
        weights[BASE_DEGRADATIONS.index(degradation)] = 1.0
        return weights

    used = [name for name in BASE_DEGRADATIONS if f"{name}[" in params or params.startswith(name)]
    if not used:
        weights.fill_(1.0 / len(BASE_DEGRADATIONS))
        return weights
    for name in used:
        weights[BASE_DEGRADATIONS.index(name)] = 1.0 / len(used)
    return weights


class DIV2KRestorationDataset(Dataset):
    """Pairs generated DIV2K LQ patches with their clean GT patches."""

    def __init__(
        self,
        data_root: str | Path = "data",
        split: str = "train",
        degradations: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        wanted = set(degradations or ALL_DEGRADATIONS)
        unknown = wanted.difference(ALL_DEGRADATIONS)
        if unknown:
            raise ValueError(f"Unknown degradations: {sorted(unknown)}")

        samples = [s for s in _read_metadata(self.data_root, split) if s.degradation in wanted]
        self.samples = samples[:limit] if limit is not None else samples
        if not self.samples:
            raise RuntimeError(f"No samples left after filtering split={split!r}, degradations={sorted(wanted)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        return {
            "lq": _image_to_tensor(sample.lq),
            "gt": _image_to_tensor(sample.gt),
            "degradation": torch.tensor(DEGRADATION_TO_INDEX[sample.degradation], dtype=torch.long),
            "mix_weights": _mix_weights(sample.degradation, sample.params),
            "name": sample.lq.stem,
            "degradation_name": sample.degradation,
        }

