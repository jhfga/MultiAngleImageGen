# MultiAngleImageGen

基于 Qwen-Image-Edit-2511 + LoRA 的多角度图像生成项目。

## 环境准备

### 创建 Conda 环境

```bash
conda create -n multiangle python=3.12 -y
conda activate multiangle
```

### 安装 CUDA Toolkit

```bash
conda install -c nvidia cuda-toolkit=12.4 -y
```

### 安装 PyTorch（GPU 版）

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124
```

### 安装其他依赖

```bash
pip install modelscope diffusers transformers accelerate bitsandbytes sentencepiece Pillow
```

## 模型下载

### 主模型

```bash
modelscope download --model Qwen/Qwen-Image-Edit-2511 --local_dir ./models/Qwen-Image-Edit-2511
```

模型约 57.7GB（BF16），下载耗时较长，请确保磁盘空间充足。

### LoRA

```bash
modelscope download --model fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --local_dir ./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
```

## 硬件要求

| 配置 | 显存需求 | 说明 |
|------|---------|------|
| BF16 原版 | ~30GB | 全精度加载 |
| FP8 量化 | ~15GB | 量化加载 |
| NF4 量化 + CPU卸载 | ~12GB | 4-bit量化，Text Encoder卸载CPU |

推荐 12GB 显存使用 NF4 量化 + CPU 卸载方案。
