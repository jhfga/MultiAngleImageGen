"""调用推理服务的示例脚本。"""

import time
import requests

SERVER_URL = "https://fed-steel-ahead-resulted.trycloudflare.com"  # 替换为你的公网地址，如 https://xxxx.trycloudflare.com
POLL_INTERVAL = 2  # 轮询间隔（秒）


def check_health():
    """健康检查。"""
    resp = requests.get(f"{SERVER_URL}/health")
    resp.raise_for_status()
    print(resp.json())


def submit_task(image_path: str, prompt: str) -> str:
    """提交推理任务，返回 task_id。"""
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SERVER_URL}/generate",
            files={"images": f},
            data={
                "prompt": prompt,
                "seed": 42,
                "num_inference_steps": 20,
                "guidance_scale": 5.0,
                "max_image_size": 1024,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    print(f"任务已提交: task_id={data['task_id']}, 排队位置={data['queue_position']}")
    return data["task_id"]


def submit_multi_task(image_paths: list[str], prompt: str) -> str:
    """提交多图推理任务，返回 task_id。"""
    files = [("images", open(p, "rb")) for p in image_paths]
    try:
        resp = requests.post(
            f"{SERVER_URL}/generate",
            files=files,
            data={
                "prompt": prompt,
                "seed": 42,
                "num_inference_steps": 20,
                "guidance_scale": 5.0,
                "max_image_size": 1024,
            },
        )
    finally:
        for _, f in files:
            f.close()
    resp.raise_for_status()
    data = resp.json()
    print(f"任务已提交: task_id={data['task_id']}, 排队位置={data['queue_position']}")
    return data["task_id"]


def wait_and_download(task_id: str, output_path: str = "output.png"):
    """轮询任务状态，完成后下载结果图片。"""
    while True:
        resp = requests.get(f"{SERVER_URL}/status/{task_id}")
        resp.raise_for_status()
        data = resp.json()

        if data["status"] == "waiting":
            print(f"\r排队中... 前方还有 {data['queue_position']} 人", end="", flush=True)
        elif data["status"] == "processing":
            step = data.get("current_step", 0)
            total = data.get("total_steps", 0)
            pct = data.get("progress", 0) * 100
            print(f"\r正在生成中... {step}/{total} ({pct:.1f}%)", end="", flush=True)
        elif data["status"] == "completed":
            print("\n生成完成，正在下载...")
            break
        elif data["status"] == "failed":
            print(f"\n生成失败: {data.get('error', '未知错误')}")
            return

        time.sleep(POLL_INTERVAL)

    # 下载结果
    resp = requests.get(f"{SERVER_URL}/result/{task_id}")
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"结果已保存至 {output_path}")


def generate(image_path: str, prompt: str, output_path: str = "output.png"):
    """一键提交 + 等待 + 下载。"""
    task_id = submit_task(image_path, prompt)
    wait_and_download(task_id, output_path)


if __name__ == "__main__":
    # 1. 检查服务状态
    check_health()

    # 2. 单图推理（一键完成）
    generate(
        image_path="test.png",
        prompt="<sks> front view eye-level shot medium shot",
        output_path="output_front.png",
    )

    # 3. 多图推理（按需使用）
    # generate(
    #     image_path="person1.png",  # submit_multi_task 内部处理多图
    #     prompt="<sks> front view eye-level shot medium shot",
    #     output_path="output_multi.png",
    # )

    # 4. 也可以分步调用，手动控制流程：
    # task_id = submit_task("test.png", "<sks> right side view high-angle shot close-up")
    # # ... 做其他事情 ...
    # wait_and_download(task_id, "output_side.png")
