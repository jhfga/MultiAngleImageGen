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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 安装其他依赖

```bash
pip install modelscope diffusers transformers accelerate bitsandbytes sentencepiece Pillow peft atomgit
```

## 模型下载

### 主模型（BF16 原版）

```bash
modelscope download --model Qwen/Qwen-Image-Edit-2511 --local_dir ./models/Qwen-Image-Edit-2511
```

模型约 57.7GB（BF16），下载耗时较长，请确保磁盘空间充足。

### 主模型（4-bit 量化版）

```bash
atomgit download hf_mirrors/toandev/Qwen-Image-Edit-2511-4bit -d ./models/Qwen-Image-Edit-2511-4bit
```

### 主模型（FP8 量化版）

```bash
modelscope download --model 1038lab/Qwen-Image-Edit-2511-FP8 --local_dir ./models/Qwen-Image-Edit-2511-FP8
```

### LoRA

```bash
modelscope download --model fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --local_dir ./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
```

## 使用方式

### 方式一：命令行推理

直接运行 `load_model.py`，对 `test.png` 生成 4 种角度的图片：

```bash
python load_model.py
```

可在代码中修改 `prompts` 列表和 `image_path` 来自定义输入。

### 方式二：启动推理服务

启动 HTTP 服务，用户通过接口上传图片和提示词即可获得生成结果。

#### 安装服务依赖

```bash
pip install fastapi "uvicorn[standard]"
```

#### 仅局域网访问

```bash
python server.py
```

服务监听 `http://0.0.0.0:8000`。

#### 公网访问（Cloudflare Quick Tunnel）

1. 安装 cloudflared：

```bash
# Ubuntu/Debian（推荐，速度快）
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /etc/apt/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/etc/apt/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared -y

# CentOS/RHEL
sudo yum install cloudflared -y

# Windows
winget install Cloudflare.cloudflared
```

2. 启动服务并自动开通公网隧道：

```bash
python server.py --tunnel
```

启动后会自动打印公网地址，如：

```
============================================================
  公网访问地址: https://xxxx-xxxx.trycloudflare.com
  API 文档:     https://xxxx-xxxx.trycloudflare.com/docs
============================================================
```

#### 服务启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--tunnel` | 关闭 | 启用 Cloudflare Quick Tunnel 公网访问 |

#### API 接口

**健康检查**

```
GET /health
```

**图像生成**

```
POST /generate
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `images` | File[] | 是 | - | 输入图片（支持多张） |
| `prompt` | string | 是 | - | 编辑提示词 |
| `seed` | int | 否 | 42 | 随机种子 |
| `num_inference_steps` | int | 否 | 20 | 推理步数 |
| `guidance_scale` | float | 否 | 5.0 | 引导系数 |
| `max_image_size` | int | 否 | 1024 | 最大图像边长 |

返回 `image/png` 图片字节流。

#### 调用示例

**curl**

```bash
curl -X POST http://localhost:8000/generate \
  -F "images=@test.png" \
  -F "prompt=<sks> front view eye-level shot medium shot" \
  -o output.png
```

**Python**

```bash
pip install requests
python client_example.py
```

详见 `client_example.py`，包含单图推理、多图推理、健康检查的完整示例。
