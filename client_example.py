"""调用推理服务的示例脚本。"""

import os
import time
import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")  # 替换为你的服务地址
API_KEY = os.environ.get("API_KEY", "替换为启动时生成的访问密钥")
POLL_INTERVAL = 2  # 轮询间隔（秒）


def check_health():
    """健康检查（无需密钥）。"""
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
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "max_image_size": 1024,
            },
            params={"key": API_KEY},
        )
    resp.raise_for_status()
    data = resp.json()
    print(f"任务已提交: task_id={data['task_id']}, 排队位置={data['queue_position']}")
    return data["task_id"]


def wait_and_download(task_id: str, output_path: str = "output.png"):
    """轮询任务状态，完成后下载结果图片。"""
    while True:
        resp = requests.get(f"{SERVER_URL}/status/{task_id}", params={"key": API_KEY})
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
    resp = requests.get(f"{SERVER_URL}/result/{task_id}", params={"key": API_KEY})
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"结果已保存至 {output_path}")


def generate(image_path: str, prompt: str, output_path: str = "output.png"):
    """一键提交 + 等待 + 下载。"""
    task_id = submit_task(image_path, prompt)
    wait_and_download(task_id, output_path)


if __name__ == "__main__":
    generate(
        image_path="test.png",
        prompt="<sks> right side view low-angle shot medium shot",
        output_path="output_front.png",
    )
