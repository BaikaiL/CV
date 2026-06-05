# 深度学习方法版：Prompt-LoRA Restormer

本目录实现课程项目的版本 B，用同一套 DIV2K 人工退化数据训练轻量图像复原网络。

## 文件说明

- `dataset.py`：读取 `data/degradation_metadata.csv`，自动配对 LQ/GT，并返回退化类型与混合退化 prompt 权重。
- `model.py`：轻量 Restormer 风格网络，包含 Patch Embedding、Transformer Encoder/Decoder、退化 Prompt 和 LoRA Attention。
- `prompt_module.py`：为 lowlight、blur、jpeg、haze 设置可学习 prompt，mixed 使用多 prompt 加权。
- `lora.py`：Q/V 投影上的低秩 LoRA 适配器，并支持只训练 LoRA + Prompt + 输出层。
- `losses.py`：`L1 + 0.1 * SSIMLoss + 0.05 * EdgeLoss`。
- `train.py`：训练与验证入口，支持 baseline/prompt/lora/prompt_lora 消融。
- `infer.py`：加载 checkpoint，对单张图片或目录推理。

## 训练示例

快速冒烟训练：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 1 --batch-size 2 --train-limit 20 --valid-limit 5 --num-workers 0 --amp
```

正式训练建议：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp
```

消融实验：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant baseline --epochs 50 --batch-size 4 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt --epochs 50 --batch-size 4 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant lora --epochs 50 --batch-size 4 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp --freeze-backbone
```

## 推理示例

```powershell
.\.venv\Scripts\python.exe -m deep.infer --checkpoint results/deep/prompt_lora_xxx/best.pth --input data/valid_LQ/mixed --output results/deep_infer/mixed --degradation mixed
```

输出 checkpoint 和 TensorBoard 日志默认保存在 `results/deep/`。

