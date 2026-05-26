# 推理服务计划

## 摘要

新建一个 `server.py` 文件，基于 FastAPI + uvicorn 启动公网可访问的推理服务。服务启动时预热模型，用户通过 HTTP 接口上传图片和提示词，服务返回生成结果。推理过程串行排队，图片预处理等可并行部分尽量并行化。

## 当前状态分析

- 项目仅有一个 `load_model.py`，包含 `load_model_4bit()` 和 `run_inference()` 两个函数
- 无任何 Web 服务代码、依赖管理文件或配置文件
- 模型使用 `enable_model_cpu_offload()`，推理不可并行（GPU 显存限制）
- `run_inference()` 中图片加载和缩放逻辑可独立于推理，适合并行化

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| Web 框架 | FastAPI | 原生 async、自动 OpenAPI 文档、类型校验、生态成熟 |
| ASGI 服务器 | uvicorn | FastAPI 标准搭配，生产级性能 |
| 图片上传方式 | multipart/form-data | 适合文件上传，前端调用简单 |
| 推理排队机制 | `asyncio.Lock` | 轻量，确保同一时刻只有一个推理任务占用模型 |
| 并行化策略 | `asyncio.to_thread` + 线程池 | 图片预处理（加载、缩放、编码）在线程池并行执行，不阻塞事件循环 |
| 返回格式 | 直接返回 PNG 图片字节流 | 最简单直接，前端可用 `<img>` 标签直接展示 |

## 新建文件

### `server.py` — 推理服务主文件

**核心结构：**

```
server.py
├── 全局变量：pipe (模型实例), inference_lock (推理锁)
├── startup 事件：预热模型
├── POST /generate — 核心推理接口
├── GET /health — 健康检查
└── main() — uvicorn 启动入口
```

**接口设计：**

#### `POST /generate`

- **请求**：multipart/form-data
  - `image`: 上传的图片文件（支持多张，用 `images` 字段，FastAPI `List[UploadFile]`）
  - `prompt`: 编辑提示词（str）
  - `seed`: 随机种子（int，默认 42）
  - `num_inference_steps`: 推理步数（int，默认 20）
  - `guidance_scale`: 引导系数（float，默认 5.0）
  - `max_image_size`: 最大图像边长（int，默认 1024）
- **响应**：`image/png` 直接返回图片字节流
- **错误**：JSON 格式错误信息，HTTP 422/500

#### `GET /health`

- **响应**：`{"status": "ok", "model_loaded": true/false}`

**并行化策略：**

1. **图片预处理并行**：多张图片的加载+缩放使用 `asyncio.gather(*[asyncio.to_thread(load_and_resize, f, max_size) for f in files])` 并行执行
2. **推理串行**：通过 `async with inference_lock:` 确保同一时刻只有一个推理任务
3. **响应编码并行**：推理完成后，图片编码为 PNG 字节可在后台线程执行，不阻塞下一个请求的接收

**预热流程：**

- 在 FastAPI `lifespan` 上下文管理器中调用 `load_model_4bit()`
- 加载完成后打印日志，服务开始接受请求
- 若加载失败，服务不启动

**关键实现细节：**

- 复用 `load_model.py` 中的 `load_model_4bit()` 加载模型
- 推理部分不直接调用 `run_inference()`（它会保存文件），而是内联推理逻辑，将结果图片编码为字节流直接返回，避免不必要的磁盘 I/O
- 使用 `asyncio.to_thread` 将同步的模型推理调用包装为异步，不阻塞事件循环（其他请求可正常排队等待）
- 设置合理的请求超时和错误处理

## 修改文件

### `load_model.py` — 无需修改

现有函数签名和逻辑完全可复用，`server.py` 直接 import 使用。

## 依赖

需新增的 pip 依赖：
- `fastapi`
- `uvicorn[standard]`

## 验证步骤

1. 启动服务：`python server.py`
2. 确认控制台输出模型加载完成日志
3. 访问 `http://localhost:8000/health` 确认 `model_loaded: true`
4. 使用 curl 测试推理：
   ```bash
   curl -X POST http://localhost:8000/generate \
     -F "images=@test.png" \
     -F "prompt=<sks> front view eye-level shot medium shot" \
     -o output.png
   ```
5. 确认返回的 `output.png` 可正常打开
6. 同时发送两个请求，确认第二个请求排队等待而非报错
