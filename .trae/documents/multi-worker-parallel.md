# 多模型实例并行处理

## 摘要

将当前单 worker 串行推理改为 N 个 worker 并行推理。用户请求到达后先在 CPU 做预处理进入队列，N 个独立加载的模型实例各自从队列领任务处理，完成后返回结果。

## 现状分析

当前 `server.py` 的 `TaskManager` 只有一个 `_worker` 协程，从 `asyncio.Queue` 中逐个取任务串行执行。`_run_inference` 使用全局 `pipe` 对象。即使有多个请求排队，也只有一个 GPU 推理在跑。

**关键约束**：用户选择单 GPU + 每个实例独立加载。这意味着 N 个模型实例共享同一张 GPU 的显存，推理时 GPU 串行执行（CUDA 本质上是串行队列），但 N 个 worker 可以让一个实例在编码 PNG / 释放资源时另一个实例立即开始推理，减少空闲间隙。

## 方案设计

### 核心改动：`TaskManager` 支持多 worker

1. **`__init__` 接收 `num_workers` 参数**，启动对应数量的 `_worker` 协程
2. **每个 worker 持有自己的 `pipe` 实例**（独立加载，独立显存）
3. **共享同一个 `asyncio.Queue`**，谁空闲谁领任务
4. **`_run_inference` 接收 `pipe` 参数**而非使用全局变量

### 数据结构改动

```python
class TaskManager:
    def __init__(self, num_workers: int = 1):
        self._tasks: dict[str, Task] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers_started = False
        self._num_workers = num_workers
        self._pipes: list = []  # 每个 worker 的 pipe 实例
```

### 启动流程改动

- `lifespan` 中循环调用 `load_model_4bit()` N 次，每次加载一个独立实例
- 打印每个实例的加载进度
- 启动 N 个 worker 协程

### `_worker` 改动

- 接收 `worker_id` 和 `pipe` 参数
- 从共享队列取任务，用自己的 `pipe` 执行推理
- `_run_inference` 改为接收 `pipe` 参数

### 启动参数

新增 `--workers` 参数（默认 1），指定模型实例数量。

### `/health` 改动

返回 `model_loaded` 改为返回 `workers` 数量信息。

## 文件改动清单

| 文件 | 改动内容 |
|------|----------|
| `server.py` | 1. `TaskManager.__init__` 接收 `num_workers`，存储 `_pipes` 列表<br>2. `start_worker` 启动 N 个 worker，每个传入对应 pipe<br>3. `_worker` 接收 `worker_id` 和 `pipe` 参数<br>4. `_run_inference` 接收 `pipe` 参数，不再使用全局 `pipe`<br>5. `lifespan` 循环加载 N 个模型实例<br>6. 删除全局 `pipe` 变量<br>7. 新增 `--workers` 启动参数<br>8. `/health` 返回 workers 信息<br>9. 启动打印信息包含 workers 数量 |

## 假设与决策

1. **决策**：每个 worker 独立加载模型，占 N 倍显存（用户确认）
2. **决策**：所有 worker 共享同一张 GPU（`cuda:0`），单 GPU 场景
3. **假设**：用户 GPU 显存足够容纳 N 个 4-bit 模型实例（4-bit 约 15-20GB/个）
4. **决策**：不修改 `load_model.py`，加载逻辑不变
5. **决策**：`_update_positions` 逻辑不变，多 worker 场景下同样适用

## 验证步骤

1. `python server.py --workers 2` 启动服务，确认两个模型实例加载成功
2. 提交 3 个任务，观察是否 2 个并行处理、1 个排队
3. `/status` 接口正确反映排队位置和进度
4. `python server.py --workers 1` 退化为原有串行行为
5. `python -m py_compile server.py` 语法检查通过
