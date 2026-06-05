# 基于 DIV2K 的多退化图像复原与质量增强系统

数据集：https://github.com/Benjamin-Wegener/DIV2K#

本项目是一个课程项目，目标是在统一的 DIV2K 数据集和人工退化流程上，实现多退化图像复原与质量增强。当前项目重点完成了：

- DIV2K 高清图像人工退化数据生成
- 深度学习方法版：Prompt-LoRA Restormer
- PSNR / SSIM / LPIPS 指标扩展
- 训练过程曲线可视化
- 原图、退化图、未训练模型、训练后模型的复原对比图

项目整体流程：

```text
DIV2K 高清图像
   ↓
人工合成退化
   ↓
低光 / 运动模糊 / JPEG / 雾霾 / 混合退化
   ↓
Prompt-LoRA Restormer 训练
   ↓
PSNR / SSIM 指标验证
   ↓
训练曲线与复原对比图可视化
```

## 1. 环境配置

建议在虚拟环境中安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

确认 PyTorch 和 CUDA：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

当前 `requirements.txt` 使用 CUDA 12.4 PyTorch 轮子。如果机器没有 NVIDIA GPU，需要改装 CPU 版 PyTorch。

## 2. 数据集划分

原始 DIV2K 数据放在：

```text
data/
├── DIV2K_train_HR/
└── DIV2K_valid_HR/
```

划分方式：

```text
训练集：DIV2K_train_HR，共 800 张原始高清图
验证集：DIV2K_valid_HR，共 100 张原始高清图
```

人工退化生成后，项目使用：

```text
data/
├── train_GT/       # 训练 GT patch
├── valid_GT/       # 验证 GT patch
├── train_LQ/       # 训练退化图像
├── valid_LQ/       # 验证退化图像
└── degradation_metadata.csv
```

退化类型：

```text
lowlight：低光照
blur：运动模糊
jpeg：JPEG 压缩噪声
haze：雾霾
mixed：混合退化
```

已生成数据规模：

```text
train_GT：6400 张 256×256 patch
valid_GT：100 张验证 patch
train_LQ：5 类退化，每类 6400 张
valid_LQ：5 类退化，每类 100 张
```

其中 `degradation_metadata.csv` 记录每张 LQ 图像对应的 GT、退化类型和退化参数，是深度学习数据集读取的主要依据。

## 3. 人工退化数据生成

如果需要重新生成人工退化数据，可以运行：

```powershell
.\.venv\Scripts\python.exe degradations\generate_dataset.py --data-root data --patches-per-train-image 8 --patches-per-valid-image 1 --valid-center-crop --overwrite
```

生成结果包括：

```text
clean.png → lowlight.png
clean.png → blur.png
clean.png → jpeg.png
clean.png → haze.png
clean.png → mixed.png
```

退化实现位于：

```text
degradations/
├── lowlight.py
├── motion_blur.py
├── jpeg.py
├── haze.py
└── generate_dataset.py
```

## 4. 训练

```text
默认 epochs = 12
默认 dim = 24
默认每轮训练样本 = 1200
默认验证样本 = 40
默认每 2 个 epoch 验证一次
默认支持早停
新训练会保存 metrics_history.csv
```

推荐训练命令：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 12 --batch-size 4 --num-workers 4 --amp
```

更快的预实验命令：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 6 --batch-size 4 --course-train-samples 600 --course-valid-samples 20 --num-workers 4 --amp
```

如果希望每轮都验证，方便画更密集的 PSNR / SSIM 曲线：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp
```

完整训练命令：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 50 --batch-size 4 --num-workers 4 --amp --full-training
```

完整训练会使用全部训练和验证数据，耗时明显更长。

## 5. 消融实验命令

可以使用同一训练规模对比 baseline、Prompt、LoRA、Prompt+LoRA：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant baseline --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp
```

如需展示 LoRA 少参数训练：

```powershell
.\.venv\Scripts\python.exe -m deep.train --variant prompt_lora --epochs 8 --batch-size 4 --course-train-samples 800 --course-valid-samples 30 --val-interval 1 --num-workers 4 --amp --freeze-backbone
```

## 6. 可视化训练结果

训练结束后，会生成类似目录：

```text
results/deep/prompt_lora_20260605_171125/
├── best.pth
├── last.pth
├── metrics_history.csv
└── tensorboard/
```

可视化命令：

```powershell
.\.venv\Scripts\python.exe -m deep.visualize_results --run-dir results\deep\prompt_lora_20260605_171125 --degradation mixed --num-samples 3
```

输出目录：

```text
results/deep/prompt_lora_20260605_171125/visualizations/
├── loss_curve.png
├── psnr_curve.png
├── ssim_curve.png
└── comparisons/
    ├── 01_mixed_xxx.png
    ├── 02_mixed_xxx.png
    └── 03_mixed_xxx.png
```

曲线图说明：

```text
loss_curve.png：每轮训练 loss 变化
psnr_curve.png：验证 PSNR 变化
ssim_curve.png：验证 SSIM 变化
```

对比图四列依次为：

```text
GT Clean：原始清晰图
LQ Degraded：人工退化图
Untrained Model：未训练模型输出
Trained Model：训练后模型复原
```

查看不同退化类型时，修改 `--degradation`：

```text
lowlight
blur
jpeg
haze
mixed
```

例如查看去雾效果：

```powershell
.\.venv\Scripts\python.exe -m deep.visualize_results --run-dir results\deep\prompt_lora_xxx --degradation haze --num-samples 3
```


## 7. 推理命令

对单张图像推理：

```powershell
.\.venv\Scripts\python.exe -m deep.infer --checkpoint results\deep\prompt_lora_xxx\best.pth --input data\valid_LQ\mixed\0801_p001.png --output results\deep_infer\0801_p001.png --degradation mixed
```

对目录推理：

```powershell
.\.venv\Scripts\python.exe -m deep.infer --checkpoint results\deep\prompt_lora_xxx\best.pth --input data\valid_LQ\mixed --output results\deep_infer\mixed --degradation mixed
```

## 8. TensorBoard

训练时会写入 TensorBoard 日志：

```text
results/deep/<run_name>/tensorboard/
```

启动 TensorBoard：

```powershell
.\.venv\Scripts\tensorboard.exe --logdir results\deep
```

然后在浏览器中查看：

```text
http://localhost:6006
```

## 9. 项目结构

```text
CV/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── DIV2K_train_HR/
│   ├── DIV2K_valid_HR/
│   ├── train_GT/
│   ├── valid_GT/
│   ├── train_LQ/
│   │   ├── lowlight/
│   │   ├── blur/
│   │   ├── jpeg/
│   │   ├── haze/
│   │   └── mixed/
│   ├── valid_LQ/
│   │   ├── lowlight/
│   │   ├── blur/
│   │   ├── jpeg/
│   │   ├── haze/
│   │   └── mixed/
│   └── degradation_metadata.csv
├── degradations/
│   ├── __init__.py
│   ├── lowlight.py
│   ├── motion_blur.py
│   ├── jpeg.py
│   ├── haze.py
│   ├── generate_dataset.py
│   └── README.md
├── deep/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── prompt_module.py
│   ├── lora.py
│   ├── losses.py
│   ├── train.py
│   ├── infer.py
│   ├── visualize_results.py
│   ├── README.md
│   └── 代码与训练逻辑总结.md
├── metrics/
│   ├── __init__.py
│   ├── psnr_ssim.py
│   └── lpips_eval.py
└── results/
    ├── deep/
    ├── deep_fast_smoke/
    └── deep_infer/
```

说明：

- `data/` 和 `results/` 数据量较大，默认被 `.gitignore` 忽略。
- `data/` 存放原始 DIV2K、GT patch 和退化数据。
- `results/` 存放训练 checkpoint、TensorBoard 日志、指标曲线和可视化对比图。


