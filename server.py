import argparse
import asyncio
import io
import contextlib
import secrets
import uuid
import time
from dataclasses import dataclass, field

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import Response

from load_model_lightning_lora import load_model_lightning_lora

# ── 任务状态 ─────────────────────────────────────────────
@dataclass
class Task:
    id: str
    status: str = "waiting"       # waiting / processing / completed / failed
    queue_position: int = 0       # 排队位置，0 表示正在处理
    progress: float = 0.0         # 推理进度 0.0 ~ 1.0
    current_step: int = 0         # 当前推理步数
    total_steps: int = 0          # 总推理步数
    created_at: float = field(default_factory=time.time)
    input_images: object = None          # 预处理后的 PIL Image
    prompt: str = ""
    seed: int = -1              # -1 表示随机种子
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    result: bytes | None = None   # PNG 字节
    error: str | None = None


class TaskManager:
    """管理推理任务队列，支持多 worker 并行推理。"""

    def __init__(self, num_workers: int = 1):
        self._tasks: dict[str, Task] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers_started = False
        self._num_workers = num_workers
        self._pipes: list = []  # 每个 worker 的 pipe 实例

    def start_worker(self):
        if not self._workers_started:
            for i, pipe in enumerate(self._pipes):
                asyncio.create_task(self._worker(i, pipe))
            self._workers_started = True

    async def _worker(self, worker_id: int, pipe):
        """后台 worker，从共享队列领任务，用自己的 pipe 执行推理。"""
        while True:
            task_id = await self._queue.get()
            task = self._tasks.get(task_id)
            if task is None:
                continue

            task.status = "processing"
            task.queue_position = 0
            task.total_steps = task.num_inference_steps

            try:
                output_image = await asyncio.to_thread(
                    _run_inference,
                    pipe,
                    task.input_images,
                    task.prompt,
                    task.seed,
                    task.num_inference_steps,
                    task.guidance_scale,
                    task,
                )
                task.progress = 1.0
                task.current_step = task.total_steps
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
task_manager = TaskManager()

# ── 认证 ─────────────────────────────────────────────────
API_KEY: str = ""  # 启动时自动生成


def verify_api_key(key: str = Query(..., description="访问密钥")):
    """验证 API Key。"""
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=403, detail="无效的访问密钥")
    return key


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
    pipe,
    input_images,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    task: Task,
) -> Image.Image:
    """执行模型推理，返回生成的 PIL Image。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # 获取最终输入图像尺寸，并指定为输出尺寸（对齐到 vae_scale_factor*2 的倍数）
    if isinstance(input_images, list):
        final_w, final_h = input_images[0].size
    else:
        final_w, final_h = input_images.size
    required_div = pipe.vae_scale_factor * 2
    final_w = round(final_w / required_div) * required_div
    final_h = round(final_h / required_div) * required_div

    def _on_step_end(pipe, step, timestep, callback_kwargs):
        """每步推理结束后更新任务进度。"""
        task.current_step = step + 1
        task.progress = round((step + 1) / num_inference_steps, 4)
        return callback_kwargs

    inputs = {
        "prompt": prompt,
        "image": input_images,
        "width": final_w,
        "height": final_h,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": 1.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
        "generator": generator,
        "callback_on_step_end": _on_step_end,
        "callback_on_step_end_tensor_inputs": [],
    }

    with torch.inference_mode():
        result = pipe(**inputs)

    return result.images[0]


# ── FastAPI 应用 ─────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型实例，关闭时清理。"""
    num_workers = task_manager._num_workers
    for i in range(num_workers):
        print(f"正在加载模型实例 {i + 1}/{num_workers}，请稍候...")
        pipe = load_model_lightning_lora()
        task_manager._pipes.append(pipe)
        print(f"模型实例 {i + 1}/{num_workers} 加载完成")
    print(f"全部 {num_workers} 个模型实例已就绪！")
    task_manager.start_worker()
    yield
    task_manager._pipes.clear()


app = FastAPI(title="Multi-Angle Image Generation Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workers": len(task_manager._pipes),
        "model_loaded": len(task_manager._pipes) > 0,
    }


@app.post("/generate")
async def generate(
    images: list[UploadFile] = File(..., description="输入图片（支持多张）"),
    prompt: str = Form(..., description="编辑提示词"),
    key: str = Depends(verify_api_key),
):
    """提交推理任务，立即返回 task_id，推理在后台排队执行。"""
    if not task_manager._pipes:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # ── 并行：读取上传文件字节（UploadFile.read 是异步方法） ──
    file_bytes_list = await asyncio.gather(*[f.read() for f in images])

    # ── 并行：图片预处理（加载+缩放） ─────────────────────
    preprocess_tasks = [
        asyncio.to_thread(_load_and_resize, fb, 1024)
        for fb in file_bytes_list
    ]
    processed_images = await asyncio.gather(*preprocess_tasks)

    # 单张图片直接传 Image，多张传 list[Image]
    input_images = processed_images[0] if len(processed_images) == 1 else processed_images

    # ── 随机种子 ──────────────────────────────────────────
    seed = secrets.randbelow(2**32)

    # ── 创建任务并入队 ────────────────────────────────────
    task = Task(
        id=uuid.uuid4().hex[:12],
        input_images=input_images,
        prompt=prompt,
        seed=seed,
    )
    task_manager.submit(task)

    return {"task_id": task.id, "status": "waiting", "queue_position": task.queue_position, "seed": seed}


@app.get("/status/{task_id}")
async def get_status(task_id: str, key: str = Depends(verify_api_key)):
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
    if task.status == "processing":
        resp["progress"] = task.progress
        resp["current_step"] = task.current_step
        resp["total_steps"] = task.total_steps
    if task.status == "failed":
        resp["error"] = task.error
    return resp


@app.get("/result/{task_id}")
async def get_result(task_id: str, key: str = Depends(verify_api_key)):
    """获取已完成的推理结果图片。"""
    task = task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status == "waiting" or task.status == "processing":
        raise HTTPException(status_code=202, detail="任务尚未完成")
    if task.status == "failed":
        raise HTTPException(status_code=500, detail=task.error)

    return Response(content=task.result, media_type="image/png")


# ── 启动入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Multi-Angle Image Generation Service")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="监听端口 (默认: 8001)")
    parser.add_argument("--workers", type=int, default=1, help="模型实例数量 (默认: 1)")
    args = parser.parse_args()

    # 设置 worker 数量
    task_manager._num_workers = args.workers

    # 自动生成 API Key
    API_KEY = secrets.token_urlsafe(24)

    print(f"\n{'='*60}")
    print(f"  服务已启动，监听端口: {args.port}")
    print(f"  模型实例: {args.workers}")
    print(f"  访问密钥: {API_KEY}")
    print(f"")
    print(f"  本机访问:")
    print(f"    http://127.0.0.1:{args.port}/docs?key={API_KEY}")
    print(f"")
    print(f"  远程访问（通过 SSH 端口转发）:")
    print(f"    1. 在本地终端运行:")
    print(f"       ssh -p <SSH端口> -L {args.port}:127.0.0.1:{args.port} root@<服务器地址>")
    print(f"    2. 然后在本地浏览器访问:")
    print(f"       http://127.0.0.1:{args.port}/docs?key={API_KEY}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)
