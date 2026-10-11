<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/role-set/readme.png" alt="role-set" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/role-set"><img src="https://codecov.io/gh/mcp-tool-shop-org/role-set/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/role-set/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**原型，以开放方式开发。** 仅供参考和实验：接口和数据结构会不经通知地发生变化。

角色集计划定义角色塑造项，编写作者应编写的部分，并记录每个项通过了哪些关卡。它是一个生成器，而不是一个裁判。一个好的验证器存在于角色包中，而角色包是由人类编写的。此程序编写的任何内容都不会包含在训练集中。每个项都会经过研发的关键检查和第二次阅读，然后才能接近最终状态。

作者和评论者只能通过 `http://127.0.0.1` 明确的端口进行访问。任何其他访问方式都会在发送任何字节之前被拒绝。角色集绝不会租用机器，也不会触及任何 Pod。当一个运行使用 Pod 时，offrig 负责租用，而角色集只会看到环回隧道。

设计方案请参考[docs/role-set-design.md](docs/role-set-design.md)，该方案与程序保持一致。运行记录请查看[docs/runs/](docs/runs/)。

## 安装

Python 3.12，由 uv 管理。一个运行时依赖项：[aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si)，固定到 aspire-si 的一个提交版本。这些关卡会被导入，而不是复制。

```text
uv sync --frozen
```

## 命令

`role-set --help` 列出了所有命令。下面的每一行都是当前存在的命令。

`plan` 从种子生成一批数据，并为每个槽位进行填充。它不会调用任何模型。使用相同的种子会重现相同的计划。

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` 在启动任何内容之前，估算该计划所需的 Pod 小时数和费用。该数字仅为估算值。配置文件为 `two-card` 和 `frontier`。

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` 从本地 aspire-si 克隆的特定提交版本复制角色包。`pack verify` 将该副本与其清单进行比较。克隆路径不会被记录。此仓库的新克隆版本不是同步的包：`roles/verifier` 包含排除列表和源列表，而同步会填充其余部分。

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` 将经过许可检查的摘录复制到本地池中。克隆目录不会被记录。

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` 是运行器。`--mock` 运行整个流水线，不使用任何模型，也不产生任何费用，并且不会请求任何授权。模拟输出会被标记，它不是数据，并且会写入 `<out>/mock/`。调用模型的运行需要 `--endpoint` 和 `--model`，并且端点必须是环回。加载评论者或本地排练编写器也需要 `--grant` 和 `--heartbeat`。`--no-critic` 是模拟运行：项在没有第二个阅读者的情况下通过，然后 Pod 编写器不会使用本地 GPU。

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` 报告每个课程和类型的单个运行结果，包括一个未接受任何内容的运行。它读取包含 `batch-report.json` 的目录。完成的 `write` 会留下该文件。`readout` 是将其转换为报告的命令。

```text
uv run role-set readout --run out/mock
```

`critic-control` 衡量评论者对已审查项的评估。它需要授权和心跳文件。`--trait-leg` 和 `--answer-leg` 运行这两个测量。

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` 将收据记录到存储中。防火墙在入口处运行。`store status` 报告该文件。`store void` 为一轮添加一个空值。阅读器会跳过已标记为空值的一轮。它不会删除任何行。

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` 是付费启动器：一次粘贴，然后它等待环回通道，进行编写，并且可以在发生连接中断后重新启动。测试绝不会将其用于实时端点。CI 使用 `write --mock`。

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` 运行 B 类求解器，处理一个数据文件以及与其并排的 `.models.jsonl` 侧边栏。一次性提供目录中的所有数据文件：如果一个谜题的事实框架重复了之前的谜题，但名称、行顺序和方向被排除在外，则会被拒绝，并会注明是第一个。

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## 关卡

五种类型：

- **A.** 一段教学文本。没有结论。验证器运行，并且特征检查仅供参考。
- **B.** 一个有限的分配谜题。求解器检查模型。评论者的答案仅供参考。
- **C.** 一个带有植入缺陷的检查器，或者一个正确的检查器。结论和证据检查会拒绝。缺陷检查会被记录，但不会拒绝。
- **D.** 两个回合。结论和证据检查会拒绝。特征检查仅供参考。
- **E.** 一份书面测验，或者一份结论测验。书面 E 没有结论检查。结论类型的 E 检查结论和证据。

结论和证据检查会拒绝。特征、答案和缺陷检查会被记录，并且不会拒绝，直到它们的控制项通过。特征分支的整体得分为 40.3%，而目标值为 85%；最差的特征得分为 12.5%，而目标值为 70%。答案分支的得分为 62.7%，而目标值为 90%。2026-10-11 的缺陷控制中有 9 个，总共 24 个。如果根据这些数字对特征或答案进行关卡控制，将会拒绝已知为好的项。

B 求解器处理用其封闭的规则形式编写的有限分配谜题：最多 6 个实体、3 个属性和 200,000 个分配。它不会将推理与模型进行检查。使用自由文本材料的结论型 B 项超出了求解器的范围。`puzzle-check` 将该项视为超出范围，而不是通过。

存储的防火墙在摄取时拒绝保留、密封和访谈材料，并且在某个阶段拒绝 DEV 材料。没有读取保留、密封或访谈材料的路径。存储、摄取和防火墙都已构建。指导机制已设计，但从未运行。

## 运行会保留的内容

每个已接受的收据都包含计划哈希和包哈希。如果运行使用了示例，则收据还包含示例提交和排除文件哈希。如果它使用了停用规则，则收据包含该规则的哈希。未接受任何内容的运行不会有任何收据。每个编写器调用都会被保留在 `drafts.jsonl` 中，无论是否被接受。完成的编写会留下 `batch-report.json`，并且 `readout` 会报告该目录。

## 模型

编写器固定为 Qwen3-235B-A22B-Instruct-2507，AWQ 构建，Apache-2.0。`writers.json` 将其作为 `qwen3-235b-instruct-2507` 提供，修订版本为 `96548f657d772187f16e9f04bc50aec78a9f9cdd`。

评估器是 `gemma4:31b`，摘要器是 `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`，量化为 Q4_K_M，采用 Apache-2.0 许可。它在 `writers.json` 中的端点是 `http://127.0.0.1:11434`。

`writers.json` 还列出了一个仅用于预演的生成器 `qwen3:8b`，以及一个用于比较的评估器 `granite4.1:30b`（IBM Granite 4.1，Apache-2.0）。`write` 拒绝将比较评估器作为当前运行的评估器。文件名中不存在的名称将被拒绝。文件名中包含 `cloud` 的名称将被拒绝。

本地 GPU 用于评估器和预演生成器，并且仅在授权范围内使用。这些路径需要 `--grant` 和一个心跳文件，其修改时间小于 60 秒。`critic-control` 每次运行都需要这两个文件。心跳文件不会被打开。当一个运行加载了评估器时，在退出时会卸载它，包括当运行因错误而停止时。`--mock` 不加载任何内容。

## 试点项目

三个付费试点项目记录在 [docs/runs/](docs/runs/) 中。这些是本 README 中唯一提到的成本数据。它们没有生成任何用于训练集的数据。

- **试点项目 1，2026-10-11，3.52 美元。** 104 个被接受的行中没有一个是数据。研发部门在进行任何内容审查之前，将 43 个 C 类和 D 类行标记为组装错误。相关说明见 [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md)。
- **试点项目 2，2026-10-11，3.61 美元。** 研发部门对 49 个候选文本进行了关键检查：15 个通过，15 个需要进行机械修改，19 个未通过。相关说明见 [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md)。
- **试点项目 3，2026-10-11，2.80 美元。** 它没有生成任何内容。生成器的引擎在启动后停止工作。相关说明见 [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md)。

## 威胁模型

role-set 是一个本地生成器。它不是一个服务器。它不监听，也不收集遥测数据。

一个运行会访问你命名的计划、角色包和输出目录。`ingest` 和 `store` 还会访问你命名的存储文件。`write`、`pilot-auto` 和 `critic-control` 在它们不是 `--mock` 时会访问一个环回端点。`--mock` 不发送任何内容。

一个运行不会访问云模型服务、pod、租赁帐户或租赁 API 密钥。它不会读取密封行、保留集或访谈记录。`ingest` 会读取你命名的 aspire-si 克隆中的十二个公共访谈问题，以及你命名的提交版本，只是为了拒绝重复这些问题的行。这些问题不会被存储。`store status` 返回计数，而不是行。请通过 [SECURITY.md](SECURITY.md) 中的联系方式报告漏洞。

## 测试

```text
uv sync --frozen
uv run pytest
```

代码覆盖率的最低标准为 90%，无论是在 `pyproject.toml` 中，还是在持续集成 (CI) 环境中，低于此标准都会导致测试失败。上述代码覆盖率徽章显示当前的覆盖率数值。CI 还会针对一个测试环境运行 `role-set plan` 和 `puzzle-check`，且该测试环境不包含任何模型。`scripts/verify` 通过一条命令运行相同的本地检查。

该手册与 [登录页面](https://mcp-tool-shop-org.github.io/role-set/) 一起发布。本 README 可以独立使用。

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 构建
