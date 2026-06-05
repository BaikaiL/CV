# 深度学习方法版：Prompt-LoRA Restormer

本目录实现课程项目的版本 B：使用同一套 DIV2K 人工退化数据训练轻量图像复原网络。训练脚本默认采用“课程项目模式”，减少每轮训练样本、降低验证频率并支持早停，优先保证能较快得到可展示、可对比的结果。

## 文件说明

- `dataset.py`：读取 `data/degradation_metadata.csv`，自动配对 LQ/GT，并返回退化类型与混合退化 prompt 权重。
- `model.py`：轻量 Restormer 风格网络，包含 Patch Embedding、Transformer Encoder/Decoder、退化 Prompt 和 LoRA Attention。
- `prompt_module.py`：为 lowlight、blur、jpeg、haze 设置可学习 prompt，mixed 使用多 prompt 加权。
- `lora.py`：Q/V 投影上的低秩 LoRA 适配器，并支持只训练 LoRA + Prompt + 输出层。
- `losses.py`：`L1 + 0.1 * SSIMLoss + 0.05 * EdgeLoss`。
- `train.py`：训练与验证入口，支持 baseline/prompt/lora/prompt_lora 消融。
- `infer.py`：加载 checkpoint，对单张图片或目录推理。
- `visualize_results.py`：生成训练曲线和图像复原对比图。

## 省时训练策略

默认训练参数已经按课程项目调整：

- 默认模型宽度 `dim=24`，比 `dim=32` 更快。
- 默认训练轮数 `epochs=12`。
- 默认每轮随机使用 `1200` 个训练样本。
- 默认验证使用 `40` 个验证样本。
- 默认每 `2` 个 epoch 验证一次。
- 默认 PSNR 连续 `4` 次验证没有明显提升就早停。
- 新训练会保存 `metrics_history.csv`，记录每轮 loss、PSNR、SSIM。

如果要恢复完整训练，添加：

```powershell
--full-training
```

## 推荐训练命令

课程项目推荐先跑：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 12 --batch-size 4 --num-workers 4 --amp
```

如果希望更快看到结果：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 6 --batch-size 4 --course-train-samples 600 --course-valid-samples 20 --num-workers 4 --amp
```

如果继续写 50 轮，也会自动使用课程项目模式：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp
```

此时每轮默认只训练 `1200` 个样本，并且会早停。

## 可视化结果

训练完成后，先找到本次训练目录，例如：

```text
results/deep/prompt_lora_20260605_171125
```

然后运行：

```powershell
.\.venv\Scripts\python.exe -m deep.visualize_results --run-dir results\deep\prompt_lora_20260605_171125 --degradation mixed --num-samples 3
```

脚本会在 run 目录下生成：

```text
visualizations/
├── loss_curve.png
├── psnr_curve.png
├── ssim_curve.png
└── comparisons/
    ├── 01_mixed_xxx.png
    ├── 02_mixed_xxx.png
    └── 03_mixed_xxx.png
```

其中：

- `loss_curve.png`：每轮训练 loss 变化。
- `psnr_curve.png`：每次验证 PSNR 变化。
- `ssim_curve.png`：每次验证 SSIM 变化。
- `comparisons/*.png`：四列图像对比。

四列图像依次是：

```text
GT Clean 原始清晰图
LQ Degraded 人工退化图
Untrained Model 未训练模型输出
Trained Model 训练后模型复原
```

旧 run 如果没有 `metrics_history.csv`，可视化脚本会自动从 TensorBoard 日志中读取 `epoch/loss`、`valid/psnr`、`valid/ssim` 来画图。

如果要看其他退化类型，把 `--degradation` 改成：

```text
lowlight
blur
jpeg
haze
mixed
```

例如查看去雾：

```powershell
.\.venv\Scripts\python.exe -m deep.visualize_results --run-dir results\deep\prompt_lora_20260605_171125 --degradation haze --num-samples 3
```

## 消融实验

建议每个消融实验都使用较小训练量：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant baseline --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --num-workers 4 --amp
```

如需展示 LoRA 少参数训练，可以额外跑：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --num-workers 4 --amp --freeze-backbone
```

## 完整训练

如果后面想追求更好的指标，再使用完整训练：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp --full-training
```

完整训练会使用全部训练和验证数据，并每个 epoch 都验证，耗时会明显增加。

## 推理示例

```powershell
.\.venv\Scripts\python.exe -m deep.infer --checkpoint results/deep/prompt_lora_xxx/best.pth --input data/valid_LQ/mixed --output results/deep_infer/mixed --degradation mixed
```

输出 checkpoint、TensorBoard 日志、`metrics_history.csv` 默认保存在 `results/deep/`。

