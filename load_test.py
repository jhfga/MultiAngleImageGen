"""模拟真实场景的多用户随机到达负载测试。

用户按泊松过程随机到达，流量越大到达间隔越短。
每个用户提交任务后一直排队等待直到完成，无最大等待时间。

用法:
    python load_test.py --url http://127.0.0.1:8000 --key <访问密钥> --traffic 10

    traffic 表示每分钟平均到达的用户数，越大流量越高。
"""

import argparse
import asyncio
import random
import time
import statistics
from dataclasses import dataclass, field

import aiohttp


@dataclass
class UserResult:
    """单个用户请求的结果记录。"""
    user_id: int
    arrive_time: float = 0.0       # 用户到达时间
    submit_time: float = 0.0       # 任务提交成功时间
    complete_time: float = 0.0     # 任务完成时间
    wait_seconds: float = 0.0      # 总等待时间（从到达到完成）
    queue_seconds: float = 0.0     # 排队时间（从到达到开始处理）
    process_seconds: float = 0.0   # 处理时间（从开始处理到完成）
    queue_position: int = 0        # 提交时的排队位置
    status: str = "pending"        # pending / completed / failed
    error: str = ""


@dataclass
class LoadTestStats:
    """负载测试统计。"""
    results: list[UserResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    traffic: float = 0.0

    def report(self):
        completed = [r for r in self.results if r.status == "completed"]
        failed = [r for r in self.results if r.status == "failed"]

        print(f"\n{'='*60}")
        print(f"  负载测试报告")
        print(f"{'='*60}")
        print(f"  流量设置:       {self.traffic} 请求/分钟")
        print(f"  测试时长:       {self.end_time - self.start_time:.1f} 秒")
        print(f"  总请求数:       {len(self.results)}")
        print(f"  成功:           {len(completed)}")
        print(f"  失败:           {len(failed)}")

        if completed:
            waits = [r.wait_seconds for r in completed]
            queues = [r.queue_seconds for r in completed]
            processes = [r.process_seconds for r in completed]

            print(f"\n  --- 总等待时间（排队+处理）---")
            print(f"  平均:           {statistics.mean(waits):.2f} 秒")
            print(f"  中位数:         {statistics.median(waits):.2f} 秒")
            print(f"  最短:           {min(waits):.2f} 秒")
            print(f"  最长:           {max(waits):.2f} 秒")
            if len(waits) >= 2:
                print(f"  标准差:         {statistics.stdev(waits):.2f} 秒")

            print(f"\n  --- 排队时间 ---")
            print(f"  平均:           {statistics.mean(queues):.2f} 秒")
            print(f"  中位数:         {statistics.median(queues):.2f} 秒")
            print(f"  最长:           {max(queues):.2f} 秒")

            print(f"\n  --- 处理时间 ---")
            print(f"  平均:           {statistics.mean(processes):.2f} 秒")
            print(f"  中位数:         {statistics.median(processes):.2f} 秒")

            # 分时段统计
            duration = self.end_time - self.start_time
            if duration > 0:
                throughput = len(completed) / duration
                print(f"\n  吞吐量:         {throughput:.2f} 请求/秒")

            # P90 / P95
            sorted_waits = sorted(waits)
            p90_idx = int(len(sorted_waits) * 0.9)
            p95_idx = int(len(sorted_waits) * 0.95)
            print(f"\n  --- 百分位等待时间 ---")
            print(f"  P90:            {sorted_waits[min(p90_idx, len(sorted_waits)-1)]:.2f} 秒")
            print(f"  P95:            {sorted_waits[min(p95_idx, len(sorted_waits)-1)]:.2f} 秒")

        if failed:
            print(f"\n  --- 失败详情（前 10 条）---")
            for r in failed[:10]:
                print(f"  用户 {r.user_id}: {r.error}")

        print(f"{'='*60}\n")


async def simulate_user(
    session: aiohttp.ClientSession,
    user_id: int,
    base_url: str,
    api_key: str,
    test_image_bytes: bytes,
) -> UserResult:
    """模拟单个用户：提交任务 → 一直排队等待 → 下载结果。"""
    result = UserResult(user_id=user_id)
    result.arrive_time = time.monotonic()

    try:
        # 提交任务
        data = aiohttp.FormData()
        data.add_field("images", test_image_bytes, filename="test.png", content_type="image/png")
        data.add_field("prompt", "<sks> back view low-angle shot wide shot")

        async with session.post(
            f"{base_url}/generate",
            params={"key": api_key},
            data=data,
        ) as resp:
            if resp.status != 200:
                result.status = "failed"
                result.error = f"提交失败 (HTTP {resp.status}): {await resp.text()}"
                return result
            resp_data = await resp.json()
            task_id = resp_data["task_id"]
            result.queue_position = resp_data.get("queue_position", 0)

        result.submit_time = time.monotonic()

        # 轮询状态，一直等到完成或失败
        started_processing = False
        while True:
            await asyncio.sleep(2)
            async with session.get(
                f"{base_url}/status/{task_id}",
                params={"key": api_key},
            ) as resp:
                if resp.status != 200:
                    result.status = "failed"
                    result.error = f"状态查询失败 (HTTP {resp.status})"
                    return result
                status_data = await resp.json()

            if status_data["status"] == "processing" and not started_processing:
                started_processing = True
                result.queue_seconds = time.monotonic() - result.arrive_time

            if status_data["status"] == "completed":
                if not started_processing:
                    result.queue_seconds = 0.0
                break
            elif status_data["status"] == "failed":
                result.status = "failed"
                result.error = f"推理失败: {status_data.get('error', '未知')}"
                return result

        # 下载结果
        async with session.get(
            f"{base_url}/result/{task_id}",
            params={"key": api_key},
        ) as resp:
            if resp.status != 200:
                result.status = "failed"
                result.error = f"下载失败 (HTTP {resp.status})"
                return result
            await resp.read()

        result.complete_time = time.monotonic()
        result.wait_seconds = result.complete_time - result.arrive_time
        result.process_seconds = result.complete_time - result.arrive_time - result.queue_seconds
        result.status = "completed"

    except Exception as e:
        result.status = "failed"
        result.error = str(e)

    return result


async def run_load_test(
    base_url: str,
    api_key: str,
    test_image_bytes: bytes,
    duration_seconds: int,
    traffic: float,
):
    """运行负载测试。

    用户按泊松过程到达，traffic 为每分钟平均到达数。
    到达间隔服从指数分布，均值 = 60/traffic 秒。
    """
    stats = LoadTestStats()
    stats.traffic = traffic
    stats.start_time = time.monotonic()
    deadline = stats.start_time + duration_seconds

    # 指数分布的到达间隔，均值 = 60/traffic 秒
    mean_interval = 60.0 / traffic

    print(f"负载测试开始")
    print(f"  目标服务:   {base_url}")
    print(f"  流量:       {traffic} 请求/分钟")
    print(f"  平均间隔:   {mean_interval:.1f} 秒")
    print(f"  持续时间:   {duration_seconds} 秒\n")

    async with aiohttp.ClientSession() as session:
        # 健康检查
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status != 200:
                    print("错误: 服务健康检查失败")
                    return
                health = await resp.json()
                if not health.get("model_loaded"):
                    print("错误: 模型尚未加载完成")
                    return
        except Exception as e:
            print(f"错误: 无法连接服务 - {e}")
            return

        print("服务健康检查通过，开始测试...\n")

        user_counter = 0
        active_tasks: dict[int, asyncio.Task] = {}  # user_id -> Task
        next_arrival = time.monotonic()  # 下一个用户到达时间

        while time.monotonic() < deadline or active_tasks:
            now = time.monotonic()

            # 收集已完成的任务
            done_ids = [uid for uid, t in active_tasks.items() if t.done()]
            for uid in done_ids:
                task = active_tasks.pop(uid)
                r = task.result()
                stats.results.append(r)
                elapsed = now - stats.start_time
                if r.status == "completed":
                    print(
                        f"  [{elapsed:6.1f}s] 用户 {r.user_id:3d} 完成 | "
                        f"排队 {r.queue_seconds:.1f}s + 处理 {r.process_seconds:.1f}s = "
                        f"总等待 {r.wait_seconds:.1f}s"
                    )
                else:
                    print(f"  [{elapsed:6.1f}s] 用户 {r.user_id:3d} 失败 | {r.error}")

            # 在测试期间，按泊松过程生成新用户
            while now >= next_arrival and now < deadline:
                user_counter += 1
                elapsed = now - stats.start_time
                print(f"  [{elapsed:6.1f}s] 用户 {user_counter:3d} 到达")
                task = asyncio.create_task(
                    simulate_user(session, user_counter, base_url, api_key, test_image_bytes)
                )
                active_tasks[user_counter] = task

                # 下一个到达时间（指数分布）
                interval = random.expovariate(1.0 / mean_interval)
                next_arrival = now + interval

            # 测试时间结束但还有未完成任务，等待它们
            if now >= deadline and active_tasks:
                await asyncio.sleep(2)
            else:
                # 等到下一个用户到达或下一个任务完成
                wait = min(next_arrival - now, 2.0) if now < deadline else 2.0
                if wait > 0:
                    await asyncio.sleep(max(wait, 0.1))

    stats.end_time = time.monotonic()
    stats.report()


def main():
    parser = argparse.ArgumentParser(description="多用户随机到达负载测试")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="服务地址（SSH 端口转发后默认 http://127.0.0.1:8000）")
    parser.add_argument("--key", required=True, help="访问密钥")
    parser.add_argument("--duration", type=int, default=600, help="测试时长（秒），默认 600（10 分钟）")
    parser.add_argument("--traffic", type=float, default=6.0, help="流量：每分钟平均到达用户数，默认 6")
    parser.add_argument("--image", default="test.png", help="测试图片路径，默认 test.png")
    args = parser.parse_args()

    try:
        with open(args.image, "rb") as f:
            test_image_bytes = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到测试图片 {args.image}")
        return

    asyncio.run(
        run_load_test(
            base_url=args.url,
            api_key=args.key,
            test_image_bytes=test_image_bytes,
            duration_seconds=args.duration,
            traffic=args.traffic,
        )
    )


if __name__ == "__main__":
    main()
