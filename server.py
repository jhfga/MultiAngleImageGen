import argparse
import asyncio
import io
import contextlib
import subprocess
import sys
import re
import uuid
import time
from dataclasses import dataclass, field

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from load_model import load_model_4bit

# ── 任务状态 ─────────────────────────────────────────────
@dataclass
class Task:
    id: str
    status: str = "waiting"       # waiting / processing / completed / failed
    queue_position: int = 0       # 排队位置，0 表示正在处理
    created_at: float = field(default_factory=time.time)
    input_images: object = None          # 预处理后的 PIL Image
    prompt: str = ""
    seed: int = 42
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    result: bytes | None = None   # PNG 字节
    error: str | None = None


class TaskManager:
    """管理推理任务队列，串行执行推理，支持查询排队位置。"""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_started = False

    def start_worker(self):
        if not self._worker_started:
            asyncio.create_task(self._worker())
            self._worker_started = True

    async def _worker(self):
        """后台 worker，逐个处理队列中的任务。"""
        while True:
            task_id = await self._queue.get()
            task = self._tasks.get(task_id)
            if task is None:
                continue

            task.status = "processing"
            task.queue_position = 0

            try:
                output_image = await asyncio.to_thread(
                    _run_inference,
                    task.input_images,
                    task.prompt,
                    task.seed,
                    task.num_inference_steps,
                    task.guidance_scale,
                )
                task.result = await asyncio.to_thread(_image_to_png_bytes, output_image)
                task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
            finally:
                # 释放预处理图片内存
                task.input_images = None
                self._update_positions()

    def _update_positions(self):
        """重新计算排队中任务的队列位置。"""
        position = 0
        for task_id in list(self._queue._queue):  # type: ignore
            task = self._tasks.get(task_id)
            if task and task.status == "waiting":
                position += 1
                task.queue_position = position

    def submit(self, task: Task) -> str:
        """提交任务到队列。"""
        self._tasks[task.id] = task
        self._queue.put_nowait(task.id)
        # 计算当前排队人数（不含自己，因为自己刚入队）
        waiting_count = sum(1 for t in self._tasks.values() if t.status == "waiting")
        task.queue_position = waiting_count
        return task.id

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)


# ── 全局状态 ──────────────────────────────────────────────
pipe = None
task_manager = TaskManager()


# ── 图片预处理（CPU 密集，可并行） ───────────────────────
def _load_and_resize(image_bytes: bytes, max_size: int) -> Image.Image:
    """从字节流加载图片并缩放，返回 PIL Image。"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)))
    return img


def _image_to_png_bytes(img: Image.Image) -> bytes:
    """将 PIL Image 编码为 PNG 字节流。"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── 推理逻辑（不写磁盘，直接返回 Image） ─────────────────
def _run_inference(
    input_images,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    """执行模型推理，返回生成的 PIL Image。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    inputs = {
        "prompt": prompt,
        "image": input_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
        "generator": generator,
    }

    with torch.inference_mode():
        result = pipe(**inputs)

    return result.images[0]


# ── FastAPI 应用 ─────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热模型，关闭时清理。"""
    global pipe
    print("正在加载模型，请稍候...")
    pipe = load_model_4bit()
    print("模型加载完成，服务已就绪！")
    task_manager.start_worker()
    yield
    pipe = None


app = FastAPI(title="Multi-Angle Image Generation Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}


@app.post("/generate")
async def generate(
    images: list[UploadFile] = File(..., description="输入图片（支持多张）"),
    prompt: str = Form(..., description="编辑提示词"),
    seed: int = Form(42),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(5.0),
    max_image_size: int = Form(1024),
):
    """提交推理任务，立即返回 task_id，推理在后台排队执行。"""
    if pipe is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # ── 并行：读取上传文件字节（UploadFile.read 是异步方法） ──
    file_bytes_list = await asyncio.gather(*[f.read() for f in images])

    # ── 并行：图片预处理（加载+缩放） ─────────────────────
    preprocess_tasks = [
        asyncio.to_thread(_load_and_resize, fb, max_image_size)
        for fb in file_bytes_list
    ]
    processed_images = await asyncio.gather(*preprocess_tasks)

    # 单张图片直接传 Image，多张传 list[Image]
    input_images = processed_images[0] if len(processed_images) == 1 else processed_images

    # ── 创建任务并入队 ────────────────────────────────────
    task = Task(
        id=uuid.uuid4().hex[:12],
        input_images=input_images,
        prompt=prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    task_manager.submit(task)

    return {"task_id": task.id, "status": "waiting", "queue_position": task.queue_position}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态和排队位置。"""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    resp = {
        "task_id": task.id,
        "status": task.status,
    }
    if task.status == "waiting":
        resp["queue_position"] = task.queue_position
    if task.status == "failed":
        resp["error"] = task.error
    return resp


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """获取已完成的推理结果图片。"""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status == "waiting" or task.status == "processing":
        raise HTTPException(status_code=202, detail="任务尚未完成")
    if task.status == "failed":
        raise HTTPException(status_code=500, detail=task.error)

    return Response(content=task.result, media_type="image/png")


# ── Cloudflare Quick Tunnel ──────────────────────────────
def _start_tunnel(port: int) -> subprocess.Popen:
    """启动 cloudflared quick tunnel，返回子进程对象。"""
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # 等待输出中出现公网 URL
    url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    public_url = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break
    if public_url:
        print(f"\n{'='*60}")
        print(f"  公网访问地址: {public_url}")
        print(f"  API 文档:     {public_url}/docs")
        print(f"{'='*60}\n")
    else:
        print("警告: 未能获取 cloudflared 公网地址，请确认 cloudflared 已安装")
    return proc


# ── 启动入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Multi-Angle Image Generation Service")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--tunnel", action="store_true", help="启动 Cloudflare Quick Tunnel，使服务公网可访问")
    args = parser.parse_args()

    tunnel_proc = None
    if args.tunnel:
        tunnel_proc = _start_tunnel(args.port)

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()
