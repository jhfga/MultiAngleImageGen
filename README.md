# MultiAngleImageGen

基于 Qwen-Image-Edit-2511 + Lightning 加速 + 多角度 LoRA 的多角度图像生成项目。

提供两种模型加载方案：

| 方案 | 基座模型 | 加速方式 | GPU 要求 | 脚本 | 状态 |
|------|----------|----------|----------|------|------|
| 方案一（推荐） | 4-bit 量化 | Lightning LoRA | 无特殊要求，更通用 | `load_model_lightning_lora.py` | 已验证 |
| 方案二 | BF16 原版 | FP8+Lightning 合并权重 | 需 GPU 支持 FP8 | `load_model_fp8_lightning.py` | 开发中 |

当前服务（`server.py`）使用方案一。

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

### 共享资源（两种方案都需要）

```bash
# 多角度 LoRA
modelscope download --model fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA --local_dir ./models/Qwen-Image-Edit-2511-Multiple-Angles-LoRA
```

### 方案一：4-bit 基座 + Lightning LoRA（推荐）

```bash
# 4-bit 量化基座模型
atomgit download hf_mirrors/toandev/Qwen-Image-Edit-2511-4bit -d ./models/Qwen-Image-Edit-2511-4bit
```

```bash
# Lightning 4步加速 LoRA（bf16）
# 安装 huggingface_hub（如未安装）
pip install huggingface_hub

# 设置 HuggingFace 国内镜像
# Linux:
export HF_ENDPOINT=https://hf-mirror.com
# Windows:
set HF_ENDPOINT=https://hf-mirror.com

hf download lightx2v/Qwen-Image-Edit-2511-Lightning Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors --local-dir ./models/Qwen-Image-Edit-2511-Lightning
```

### 方案二：BF16 基座 + FP8+Lightning 合并权重（需 GPU 支持 FP8）

```bash
# BF16 原版基座模型（约 57.7GB）
modelscope download --model Qwen/Qwen-Image-Edit-2511 --local_dir ./models/Qwen-Image-Edit-2511
```

```bash
# FP8+Lightning 4步加速合并权重
# 安装 huggingface_hub（如未安装）
pip install huggingface_hub

# 设置 HuggingFace 国内镜像
# Linux:
export HF_ENDPOINT=https://hf-mirror.com
# Windows:
set HF_ENDPOINT=https://hf-mirror.com

hf download lightx2v/Qwen-Image-Edit-2511-Lightning qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors --local-dir ./models/Qwen-image-edit-2511-fp8-4steps
```

> **注意**：方案二尚在开发中，FP8 合并权重文件仅适用于 LightX2V 和 ComfyUI，不能在 diffusers 中使用。

## 使用方式

### 方式一：命令行推理

**方案一（4-bit + Lightning LoRA）：**

```bash
python load_model_lightning_lora.py
```

**方案二（FP8 + Lightning）：**

```bash
python load_model_fp8_lightning.py
```

对 `test.png` 生成 4 种角度的图片，可在代码中修改 `prompts` 列表和 `image_path` 来自定义输入。

### 方式二：启动推理服务

启动 HTTP 服务，用户通过接口上传图片和提示词即可获得生成结果。

#### 安装服务依赖

```bash
pip install fastapi "uvicorn[standard]"
```

#### 启动服务

```bash
python server.py
```

启动后会自动生成访问密钥并打印：

```
============================================================
  服务已启动，监听端口: 8001
  模型实例:  1
  访问密钥:  aB3dEfGhJkLmNoPqRsTuVwXy

  本机访问:
    http://127.0.0.1:8001/docs?key=aB3dEfGhJkLmNoPqRsTuVwXy

  远程访问（通过 SSH 端口转发）:
    1. 在本地终端运行:
       ssh -p <SSH端口> -L 8001:127.0.0.1:8001 root@<服务器地址>
    2. 然后在本地浏览器访问:
       http://127.0.0.1:8001/docs?key=aB3dEfGhJkLmNoPqRsTuVwXy
============================================================
```

#### 远程访问服务

由于服务器通常无法直接通过 IP 访问，需要通过 **SSH 端口转发** 将远程服务映射到本地。

**步骤：**

1. 在远程服务器上启动服务（建议用 `nohup` 后台运行）：

```bash
nohup python server.py &
```

2. 在本地电脑终端建立 SSH 隧道：

```bash
ssh -p <SSH端口> -L 8001:127.0.0.1:8001 root@<服务器地址>
```

例如，SSH 连接方式为 `ssh -p 43965 root@cmsn7b0ulygezvd4snow.deepln.com`，则运行：

```bash
ssh -p 43965 -L 8001:127.0.0.1:8001 root@cmsn7b0ulygezvd4snow.deepln.com
```

3. 保持 SSH 连接不中断，在本地浏览器访问：

```
http://127.0.0.1:8001/docs?key=<访问密钥>
```

> **原理**：`-L 8001:127.0.0.1:8001` 表示将本地 8001 端口转发到远程机器的 127.0.0.1:8001，所有对本地 8001 端口的请求都会通过 SSH 隧道安全地传递到远程服务。

#### 服务启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8001` | 监听端口 |
| `--workers` | `1` | 模型实例数量，增加可提高并发处理能力 |

#### 启动示例

**单模型实例**（默认，适合单 GPU）

```bash
python server.py
```

**多模型实例**（多 GPU 或大显存，提高并发吞吐）

```bash
# 2 个模型实例，并行处理 2 个任务
python server.py --workers 2

# 4 个模型实例
python server.py --workers 4
```

> 注意：每个模型实例会占用一份显存，请根据 GPU 显存大小合理设置 `--workers`。

#### API 接口

所有接口（`/health` 除外）需通过查询参数 `key` 传递访问密钥进行认证。

**健康检查**（无需密钥）

```
GET /health
```

**图像生成**

```
POST /generate?key=<访问密钥>
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `images` | File[] | 是 | 输入图片（支持多张） |
| `prompt` | string | 是 | 编辑提示词 |

> 推理步数固定为 4（Lightning LoRA 加速），guidance_scale 固定为 1.0，最大输入分辨率固定为 1024px，随机种子每次自动生成。

**查询任务状态**

```
GET /status/{task_id}?key=<访问密钥>
```

**获取结果图片**

```
GET /result/{task_id}?key=<访问密钥>
```

#### 调用示例

**curl**

```bash
curl -X POST "http://127.0.0.1:8001/generate?key=你的访问密钥" \
  -F "images=@test.png" \
  -F "prompt=<sks> front view eye-level shot medium shot" \
  -o output.png
```

**Python**

```bash
pip install requests
```

设置环境变量后运行客户端示例：

```bash
set SERVER_URL=http://127.0.0.1:8001
set API_KEY=你的访问密钥
python client_example.py
```

详见 `client_example.py`，包含单图推理、多图推理、健康检查的完整示例。
