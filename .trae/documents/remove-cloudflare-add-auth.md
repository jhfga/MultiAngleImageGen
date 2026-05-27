# 摆脱 Cloudflare 依赖，添加密码认证

## 摘要

既然机器已可通过 SSH 远程访问，说明具备可达的 IP 地址。FastAPI 服务默认监听 `0.0.0.0:8000`，用户可直接通过 `http://<IP>:8000` 访问，无需 Cloudflare Tunnel。唯一缺失的是**认证机制**——当前 API 完全开放，任何人知道地址就能调用。

**目标**：移除 Cloudflare 依赖，添加密码认证，使用户知道地址+密码即可访问服务。

## 现状分析

- `server.py` 监听 `0.0.0.0:8000`，局域网/公网已可直接访问
- `--tunnel` 参数为可选功能，启动 Cloudflare Quick Tunnel
- **无任何认证机制**：所有 API 端点完全开放
- `client_example.py` 硬编码了 Cloudflare 地址，无认证逻辑
- SSH 可达意味着端口 22 已开放，需确保 8000 端口也开放

## 方案设计

### 认证方式：HTTP Basic Auth

选择理由：
- 与"知道地址+密码即可访问"的需求完全吻合
- 浏览器原生支持（自动弹出登录对话框）
- curl / requests 等工具原生支持（`-u user:pass` 或 `auth=` 参数）
- 实现简单，无需前端页面或 token 管理
- FastAPI 可通过依赖注入轻松实现

### 具体改动

#### 1. `server.py` — 添加 Basic Auth 认证

- 新增启动参数 `--username`（默认 `admin`）和 `--password`（默认 `changeme`）
- 使用 FastAPI 的 `HTTPBasic` 依赖实现认证
- 对所有 API 端点（`/health` 除外）添加认证保护
- `/health` 端点保持无需认证，方便监控探活
- 移除 `--tunnel` 参数和 `_start_tunnel()` 函数（Cloudflare 相关代码全部删除）

**认证实现方式**：
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    # 比较 username 和 password
```

#### 2. `client_example.py` — 支持认证和直接 IP 访问

- 将 `SERVER_URL` 改为直接 IP 地址格式（如 `http://192.168.x.x:8000`）
- 在所有请求中添加 `auth=(username, password)` 参数
- 可从环境变量或命令行参数读取地址和密码

#### 3. 清理 Cloudflare 相关代码

- 删除 `_start_tunnel()` 函数
- 删除 `--tunnel` 启动参数
- 删除 `import subprocess, re` 中仅用于 tunnel 的导入（如无其他用途）
- 删除启动入口中的 tunnel 相关逻辑

## 文件改动清单

| 文件 | 改动内容 |
|------|----------|
| `server.py` | 1. 添加 HTTP Basic Auth 认证（保护除 `/health` 外的所有端点）<br>2. 添加 `--username` / `--password` 启动参数<br>3. 删除 Cloudflare tunnel 相关代码（`_start_tunnel`、`--tunnel` 参数、tunnel 启动逻辑）<br>4. 清理不再需要的 import（`subprocess`, `re`） |
| `client_example.py` | 1. 修改 `SERVER_URL` 为直接 IP 格式<br>2. 所有请求添加 `auth=(username, password)`<br>3. 支持从环境变量读取配置 |

## 假设与决策

1. **假设**：用户的机器有固定 IP 或用户知道自己的 IP 地址（SSH 可达已隐含此条件）
2. **假设**：用户防火墙需开放 8000 端口（或用户指定的端口）
3. **决策**：`/health` 端点不设认证，方便负载均衡器/监控系统探活
4. **决策**：使用 HTTP Basic Auth 而非 API Key / JWT，因为最简单且符合"地址+密码"的交互模型
5. **决策**：完全移除 Cloudflare 代码而非保留为可选项，因为用户明确表示要摆脱依赖
6. **决策**：密码通过命令行参数传入（生产环境建议用环境变量，但此场景为个人/小团队使用，命令行参数更直观）

## 验证步骤

1. 启动服务：`python server.py --username admin --password mypass`
2. 无认证访问应返回 401：`curl http://localhost:8000/generate`
3. 带认证访问应正常：`curl -u admin:mypass http://localhost:8000/health`
4. `/health` 无需认证也能访问：`curl http://localhost:8000/health`
5. 运行 `client_example.py` 验证完整流程
6. 确认 Cloudflare 相关代码已完全移除，`--tunnel` 参数不再存在
