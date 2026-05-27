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

### 主模型（FP8 量化 + Lightning 4步加速版）

Lightning 加速通过 LoRA 实现，需要先下载 FP8 基础模型（见上方），再下载 Lightning LoRA：

```bash
# 安装 huggingface_hub（如未安装）
pip install huggingface_hub

# 设置 HuggingFace 国内镜像
# Linux:
export HF_ENDPOINT=https://hf-mirror.com
# Windows:
set HF_ENDPOINT=https://hf-mirror.com

# 下载 Lightning LoRA（4步加速）
hf download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir ./models/Qwen-Image-Edit-2511-Lightning
```

> **注意**：`lightx2v/Qwen-Image-Edit-2511-Lightning` 仓库中的 scaled FP8 单文件（`qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors`）仅适用于 LightX2V 和 ComfyUI，不能在 diffusers 中使用。在 diffusers 中应使用 Lightning LoRA 方式加载。

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

#### 启动服务

```bash
python server.py
```

启动后会自动生成访问密钥并打印：

```
============================================================
  服务已启动，监听端口: 8000
  模型实例:  1
  访问密钥:  aB3dEfGhJkLmNoPqRsTuVwXy

  本机访问:
    http://127.0.0.1:8000/docs?key=aB3dEfGhJkLmNoPqRsTuVwXy

  远程访问（通过 SSH 端口转发）:
    1. 在本地终端运行:
       ssh -p <SSH端口> -L 8000:127.0.0.1:8000 root@<服务器地址>
    2. 然后在本地浏览器访问:
       http://127.0.0.1:8000/docs?key=aB3dEfGhJkLmNoPqRsTuVwXy
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
ssh -p <SSH端口> -L 8000:127.0.0.1:8000 root@<服务器地址>
```

例如，SSH 连接方式为 `ssh -p 43965 root@cmsn7b0ulygezvd4snow.deepln.com`，则运行：

```bash
ssh -p 43965 -L 8000:127.0.0.1:8000 root@cmsn7b0ulygezvd4snow.deepln.com
```

3. 保持 SSH 连接不中断，在本地浏览器访问：

```
http://127.0.0.1:8000/docs?key=<访问密钥>
```

> **原理**：`-L 8000:127.0.0.1:8000` 表示将本地 8000 端口转发到远程机器的 127.0.0.1:8000，所有对本地 8000 端口的请求都会通过 SSH 隧道安全地传递到远程服务。

#### 服务启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |
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

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `images` | File[] | 是 | - | 输入图片（支持多张） |
| `prompt` | string | 是 | - | 编辑提示词 |
| `seed` | int | 否 | 42 | 随机种子 |
| `num_inference_steps` | int | 否 | 20 | 推理步数 |
| `guidance_scale` | float | 否 | 5.0 | 引导系数 |
| `max_image_size` | int | 否 | 1024 | 最大图像边长 |

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
curl -X POST "http://127.0.0.1:8000/generate?key=你的访问密钥" \
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
set SERVER_URL=http://127.0.0.1:8000
set API_KEY=你的访问密钥
python client_example.py
```

详见 `client_example.py`，包含单图推理、多图推理、健康检查的完整示例。
