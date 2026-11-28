# Patch Agent

Patch Agent 是一个基于 mini-SWE-agent 的轻量 AI 软件工程助手。给它一个代码目录和任务，它可以读代码、修改文件、运行测试并保留每步的证据。长任务可以压缩旧上下文，随后按关键词找回原始日志或源码记录。中断后可以恢复会话，检查已有改动并继续。

## 安装和配置模型

需要 Python 3.10 或更高版本、Bash，以及支持工具调用的模型服务。以下命令适用于 Linux、macOS 和 WSL。

```bash
git clone https://github.com/z-xinhong/patch-agent.git
cd patch-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

第一次使用，可以从无密钥的示例文件开始。

```bash
cp -n .env.example .env
# 编辑 .env，填写模型名和该服务的 API Key。
set -a
source ./.env
set +a
```

`.env.example` 里的 OpenAI 示例用 `MSWEA_MODEL_NAME` 选择模型，用 `OPENAI_API_KEY` 提供密钥。使用 DeepSeek 时，把模型名改成你的服务支持的 `deepseek/...`，填入 `DEEPSEEK_API_KEY`。真实密钥保存在本地 `.env` 或进程环境变量里。

你也可以运行 `mini-extra config setup`，按提示保存默认模型和密钥。两个入口都读取全局配置，Linux 默认在 `~/.config/mini-swe-agent/.env`。启动信息会显示实际路径。已导出的环境变量优先于全局文件。终端可以用 `--model` 临时覆盖模型，浏览器入口用 `-c model.model_name=提供商/模型名`。

## 选择终端或浏览器

下面两种启动方式使用同一套会话引擎、工具和验收配置。将 `/path/to/project` 替换为要操作的代码目录。

| 入口 | 启动命令 | 能看到什么 |
| --- | --- | --- |
| 终端会话 | `mini --session --workspace /path/to/project -c examples/patch-agent.yaml` | 会话 ID、工具输出和证据 ID、上下文用量、搜索与回读、改动文件和验收状态。 |
| 浏览器 Workbench | `mini-extra workbench --workspace /path/to/project -c examples/patch-agent.yaml` | 会话、命令确认、上下文、文件 diff 和验收结果。打开终端显示的地址，默认是 `http://127.0.0.1:8765`。 |

终端可以加 `--task "检查测试失败的原因"` 直接发送任务，加 `--exit-immediately` 在处理完这一轮后退出。交互时可以继续输入下一条任务。要恢复终端会话，使用首次启动时打印的 ID，并带上相同的配置文件。

```bash
mini --resume SESSION_ID -c examples/patch-agent.yaml --task "检查已有改动，然后继续"
```

浏览器在会话列表里重开原会话。终端会话和浏览器会话各自保存，通过各自的入口恢复。两种方式都会请求确认 Shell 命令和写文件操作。终端也可以显式传入 `--yolo` 自动批准。普通 `mini` 仍提供上游的 Shell 交互模式。

### Agent 可以调用的工具

| 工具 | 操作 |
| --- | --- |
| `bash` | 搜索当前仓库、运行测试或执行其他 Shell 命令。 |
| `read_file` | 读取源码的指定行，记录文件路径、行号和指纹。 |
| `apply_patch` | 创建文件，或在匹配当前文件指纹和旧片段时修改内容。 |
| `search_evidence` | 按路径、关键词或命令名查找当前会话的归档。 |
| `recall_output` | 按证据 ID 和归档行号读取原始输出。 |

历史工具默认在 `evidence` 策略下可用。源码证据过期时，Agent 会收到重读提示。

## 运行完整示例

下面的订单任务使用已有的兼容性案例。先复制一份工作区，便于独立验收和重复运行。

```bash
DEMO="$(mktemp -d)"
cp -R examples/workbench_compat/project/. "$DEMO/"

mini --session --workspace "$DEMO" -c examples/patch-agent.yaml --exit-immediately --task \
  "先运行 python3 diagnostics.py，从归档日志找回 LEGACY EXPORT CONTRACT。按该约定在 orders.py 添加 export_orders，保留现有列表行为，然后验证修改。"

python3 examples/workbench_compat/acceptance.py "$DEMO"
```

也可以对另一份干净副本运行 `mini-extra workbench --workspace "$DEMO" -c examples/patch-agent.yaml`，在页面里提交同样的任务。任务中途退出后，重开原会话，让 Agent 搜索此前归档的诊断日志。验收通过后改动工作区，旧验收结果会显示为过期。 [兼容性实验说明](examples/workbench_compat/README.md)

## 项目配置和运行记录

[examples/patch-agent.yaml](examples/patch-agent.yaml) 同时适用于终端会话和 Web。两种入口先加载内置配置，然后从左到右合并每个 `-c`。可在最后加一个临时覆盖，例如 `-c agent.step_limit=15`。

| 配置 | 用法 |
| --- | --- |
| `model.model_name`、`model.model_kwargs` | 选择 LiteLLM 的模型和请求参数。使用兼容服务时，可在 `model.model_kwargs.api_base` 配置接口地址。 |
| `agent.step_limit`、`agent.cost_limit` | 每轮的调用次数上限和美元成本估算上限。 |
| `workbench.backend`、`workbench.docker_image` | 默认在本地工作区执行。要用 Docker，将后端设为 `docker` 并指定包含项目依赖的镜像。 |
| `environment.timeout` | 单条 Shell 命令的超时秒数。 |
| `workbench.context` | 可选 `original`、`window`、`evidence`。后者默认预留 4,096 输出 token，并按 50%、70%、85% 的输入预算阈值执行缩短、移出旧记录和摘要。设 `search_evidence: false` 可以关闭关键词搜索，保留按 ID 回读。 |
| `workbench.storage_dir` | 会话存储根目录，须在工作区外。默认是全局配置目录下的 `workbench/`，终端会话位于其中的 `.terminal/`。 |
| 工作区的 `.mini-workbench.yaml` | 写入该项目的预检与验收命令。两个入口都会读取。 |

例如，目标项目可用以下内容检查 Python 环境并在修改后运行测试。

```yaml
preflight: [python3, --version]
verify: [python3, -m, pytest, -q]
```

确认前请检查它实际要执行的命令。Workbench 会记录验证退出码、输出证据和对应的文件版本。对于终端，在会话开头复制 `Archive` 路径。对于浏览器，将鼠标悬停在上下文策略标签上查看会话目录。目录中有完整轨迹 `trajectory.json`、运行事件 `events.jsonl`、证据归档和改动记录 `changes.json`。终端还可用 `--output PATH` 导出轨迹。

```bash
mini-extra workbench-report /path/to/session/archive
```

报告会汇总 token 用量、搜索与回读次数、模型费用估算和验收结果。如果有服务商账单，可以用 `--actual-cost 0.02 --currency CNY` 记录实付金额。账单金额单独显示。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `src/minisweagent/agents/` | Agent 的任务执行循环 |
| `src/minisweagent/environments/` | 本地和容器命令执行环境 |
| `src/minisweagent/models/` | 模型接口 |
| `src/minisweagent/run/` | 终端和浏览器入口、会话、报告与评测 |
| `examples/` | 共享配置和可复现的任务示例 |

## 许可证与致谢

Patch Agent 基于 Kilian A. Lieret、Carlos E. Jimenez 及贡献者开发的 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)。项目采用 MIT 许可证，原始版权声明见 [LICENSE.md](LICENSE.md)。
