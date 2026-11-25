# Patch Agent

Patch Agent 是一个轻量的 AI 软件工程助手。给它一个问题描述和一个代码目录，它可以阅读代码、执行命令、修改文件，并按任务需要运行测试。你可以用它处理 GitHub Issue，也可以直接描述一个本地 bug 或小功能。

项目仍在开发中，基于 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)。当前从源码安装，命令行入口沿用 `mini` 和 `mini-extra`。

## 快速开始

需要 Python 3.10 或更高版本，以及所选模型服务的 API 凭据。以下命令以 Linux 和 macOS 为例。

```bash
git clone https://github.com/z-xinhong/patch-agent.git
cd patch-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

激活环境后，进入你要处理的代码目录。下面以 OpenAI 模型为例，按你实际使用的服务设置 API 密钥和模型名。

```bash
export OPENAI_API_KEY="替换成你的 API 密钥"
cd /path/to/your/project
mini --model openai/gpt-5.4-mini --task "修复订单列表为空时报错的问题，并运行相关测试"
```

任务结束后，用 `git status` 和 `git diff` 检查改动。运行 `mini --help` 可以查看配置文件、成本上限和运行轨迹等选项。

## 可选的浏览器界面

Workbench 提供本地浏览器界面，可以在一个会话里发送任务、查看命令输出和文件改动，以及查看验证结果。

```bash
mini-extra workbench --workspace /path/to/your/project -c model.model_name=openai/gpt-5.4-mini
```

启动后打开终端显示的地址，默认是 `http://127.0.0.1:8765`。

- [兼容性实验](examples/workbench_compat/README.md)包含一个 CSV 导出任务，可用来比较不同的上下文策略，并独立验证改动是否满足原有行为。
- `mini-extra workbench-report --help` 列出会话报告的选项，可汇总模型用量和验证结果。

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `src/minisweagent/agents/` | Agent 的任务执行循环 |
| `src/minisweagent/environments/` | 本地、Docker 等命令执行环境 |
| `src/minisweagent/models/` | 模型接口 |
| `src/minisweagent/run/` | CLI、浏览器界面和辅助命令 |
| `examples/` | 可独立运行的示例 |
| `tests/` | 测试 |

## 许可证与致谢

Patch Agent 基于 Kilian A. Lieret、Carlos E. Jimenez 及贡献者开发的 [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)。项目采用 MIT 许可证，原始版权声明见 [LICENSE.md](LICENSE.md)。
