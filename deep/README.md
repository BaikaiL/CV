# 深度学习方法版：Prompt-LoRA Restormer

本目录实现课程项目的版本 B，用同一套 DIV2K 人工退化数据训练轻量图像复原网络。当前训练脚本默认采用“课程项目模式”：减少每轮训练样本、减少验证频率、支持早停，优先保证能较快得到可展示、可对比的结果。

## 文件说明

- `dataset.py`：读取 `data/degradation_metadata.csv`，自动配对 LQ/GT，并返回退化类型与混合退化 prompt 权重。
- `model.py`：轻量 Restormer 风格网络，包含 Patch Embedding、Transformer Encoder/Decoder、退化 Prompt 和 LoRA Attention。
- `prompt_module.py`：为 lowlight、blur、jpeg、haze 设置可学习 prompt，mixed 使用多 prompt 加权。
- `lora.py`：Q/V 投影上的低秩 LoRA 适配器，并支持只训练 LoRA + Prompt + 输出层。
- `losses.py`：`L1 + 0.1 * SSIMLoss + 0.05 * EdgeLoss`。
- `train.py`：训练与验证入口，支持 baseline/prompt/lora/prompt_lora 消融。
- `infer.py`：加载 checkpoint，对单张图片或目录推理。

## 省时训练策略

默认训练参数已经按课程项目调整：

- 默认模型宽度 `dim=24`，比原来的 `dim=32` 更快。
- 默认训练轮数 `epochs=12`。
- 默认每轮随机使用 `1200` 个训练样本。
- 默认验证使用 `40` 个验证样本。
- 默认每 `2` 个 epoch 验证一次。
- 默认 PSNR 连续 `4` 次验证没有明显提升就早停。

这样可以避免每轮完整训练 `32000` 个 LQ patch。若要恢复完整训练，添加：

```powershell
--full-training
```

## 推荐训练命令

课程项目推荐先跑这个：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 12 --batch-size 4 --num-workers 4 --amp
```

如果希望更快看到结果：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 6 --batch-size 4 --course-train-samples 600 --course-valid-samples 20 --num-workers 4 --amp
```

如果你的原命令仍然写 50 轮，也会自动使用课程项目模式：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp
```

此时每轮默认只训练 1200 个样本，并且会早停。

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

完整训练会使用全部训练/验证数据，并每个 epoch 都验证，耗时会明显增加。

## 推理示例

```powershell
.\.venv\Scripts\python.exe -m deep.infer --checkpoint results/deep/prompt_lora_xxx/best.pth --input data/valid_LQ/mixed --output results/deep_infer/mixed --degradation mixed
```

输出 checkpoint 和 TensorBoard 日志默认保存在 `results/deep/`。

