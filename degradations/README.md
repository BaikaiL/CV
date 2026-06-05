# DIV2K Synthetic Degradation Generation

This folder generates one shared LQ/GT dataset for both the traditional and deep-learning versions of the restoration project.

## Outputs

Running `generate_dataset.py` creates:

```text
data/
+-- train_GT/
+-- valid_GT/
+-- train_LQ/
|   +-- lowlight/
|   +-- blur/
|   +-- jpeg/
|   +-- haze/
|   +-- mixed/
+-- valid_LQ/
|   +-- lowlight/
|   +-- blur/
|   +-- jpeg/
|   +-- haze/
|   +-- mixed/
+-- degradation_metadata.csv
```

## Usage

Install the needed runtime packages if they are not already available:

```bash
pip install pillow numpy
```

Quick smoke test on a tiny subset:

```bash
python degradations/generate_dataset.py --data-root data --limit-train 2 --limit-valid 1 --patches-per-train-image 2 --overwrite
```

Full generation:

```bash
python degradations/generate_dataset.py --data-root data --patches-per-train-image 8 --overwrite
```

Use `--seed` to make the synthetic degradation parameters reproducible.
