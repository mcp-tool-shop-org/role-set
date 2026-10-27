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

**原型，以开放方式开发。** 仅供参考和实验：接口和数据结构会不经通知地发生变化。在有人检查之前，角色集编写的任何内容都不会被视为训练数据。每个项目都会经过研发部门的关键检查和第二次审查，然后才能接近训练集。

角色集规划角色塑造项目，编写作者应该编写的部分，并记录每个项目通过了哪些关卡。它还在代码中构建 B 类谜题，并在模型看到它们之前对其进行验证。它是一个生成器，而不是一个裁判。一个好的角色持有者存在于角色包中，人们会编写这个角色包。

作者和评论者只能通过 `http://127.0.0.1` 处的明确端口进行访问。任何其他访问都将被拒绝，并且在发送任何字节之前。角色集绝不会租用机器，也不会触碰 Pod。当一个运行使用 Pod 时，offrig 会租用它，而角色集只会看到环回隧道。

设计文档为 [docs/role-set-design.md](docs/role-set-design.md) 和 [docs/generator-design.md](docs/generator-design.md)，并与程序保持一致。运行日志位于 [docs/runs/](docs/runs/)。

## 安装

Python 3.13+ (3.15)，由 uv 管理。一个运行时依赖项：[aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si) 1.4.0，固定到 aspire-si 提交 `388a6bc`。这些关卡是导入的，而不是复制的。

```text
uv sync --frozen
```

## 角色包

角色包是人们为某个角色编写的文本。角色集以切片的形式提供，并且不会添加任何内容。它的哈希值位于每个已接受的收据中。

- **验证器** (`roles/verifier`)。`pack sync` 从本地 aspire-si 克隆的特定提交版本中复制：课程、18 个特征、类型表、修订和规则。`pack verify` 将副本与其清单进行比较。一个新的克隆只包含源列表、排除列表和 `defects.jsonl`，即审查发现的缺陷类别目录。同步会填充其余部分。
- **倡导者** (`roles/advocate`)。一个人在这个仓库中编写的包，其中不包含任何模型文本。总监于 2026-10-14 批准了该包（在拉取请求 #56 上的提交 `ffb9546`），并且第一个批次正在构建；它直到 #56 合并后才会出现在 `main` 上。第一个批次包含三个特征：`failed-gate-first`、`exact-figures` 和 `faithful-relay`。在通过试点阅读之前，任何倡导者项目都不会被视为训练数据。

## T10 生成器

过去，B 类谜题是手工编写然后进行检查。现在，代码构建谜题并对其进行验证，然后模型才会对其进行描述。该生成器是 `role_set.generate` 包。T10，关于严格大小词（“完全”、“至少”、“最多”）的课程，是 0.2.0 中发生变化的部分。还会构建一个 T12 方案（在特定日期生效的版本）。T11 则不会。

**通过构建进行验证。** 构建器会采样一个隐藏的解决方案，添加在解决方案中为真的线索，直到答案确定，然后删除答案不需要的每个线索。求解器检查模型。然后，构建器运行研发部门的四个 T10 阅读：将“完全”读作“至少”，“完全”读作“最多”，“至少”读作“完全”，以及“最多”读作“完全”。只有当这些阅读中的一个应用于特征线索时，才会改变答案，谜题才会被保留。如果谜题在其他计数线索上通过，则它是缺陷类别 `trait_not_tested`，并且会被拒绝。

**单元格映射。** 每个包、问题和种子都映射到结构单元格：实体计数 `n`（3 到 5），计数词 `op`，`k`（1 到 n-1），以及 `ifs`，即 `if` 线索的数量（0 到 2）。集合问题使用所有三个计数词。声明问题仅使用“完全”，因为“至少”和“最多”无法改变声明的结论。可用的单元格会被探测一次，并在 `t10_cells.json` 中提交。

**声明轴 `if_shape`。** 声明单元格还命名其 `if` 线索的形状。有七种形状：`none` 表示没有 `if`；`pos` 或 `neg` 表示一个 `if`，由其结论的极性决定；以及 `chain`、`shared_consequent`、`shared_antecedent` 或 `disjoint` 表示两个。如果构建的简化谜题失去了其形状，则它会在同一个种子中重试。

**分配器。** 对于一个问题和一个值计数，一个包含 20 个种子的文件会在构建任何内容之前分配一次，通过最小成本流进行分配。每个包的 20 个种子使用 20 个不同的单元格。一个没有 `if` 线索的单元格在一个文件中最多使用两次（零-如果上限）。其他访问优先使用在调整种子中产生更多不同骨架的单元格。这些权重对于两个值和三个值的包是分开的。

**多样性下限。** 在每个层级（集合和声明，两个和三个值）：至少 75% 的不同规范骨架，没有骨架出现超过 3 次，并且在一个包的文件中没有骨架出现两次。这些下限是在锁定的种子上进行判断的。在锁定的种子 101 到 120 上，三个值的声明达到了 120 个中的 118 个不同的，并且最高的重复次数为 2。

**已提交的记录。** `t10_record.json` 包含 1,280 个构建（种子 1–20、61–80、81–100 和 101–120，每个包，两个问题），每个构建都有其单元格、`if_shape`、尝试次数和规范骨架。它包含生成器源代码、单元格表、容量表和包数据上的 sha256 校验和。一个测试确保该文件列表在角色集自己的导入下是封闭的。正常的 CI 测试作业会在几分钟内检查哈希值和记录中的下限。对每个谜题进行详尽的重建，作为 `sweep` 作业运行，仅在 `workflow_dispatch` 上运行，并且如果与记录有任何差异，则会失败。

**容量门控和剩余空间。** `scripts/t10_usable_cells.py --check` 报告每个层级的供应、需求和剩余空间，并在无法放置文件时失败。使用 `if_shape`，三个值的包总共有 61 个可用的声明单元格，而之前只有 27 个。没有包需要零-`if` 访问，并且一个测试放置了十二个三个值的包（六个真实的包和六个克隆），并且在一个文件中没有重复。

## 命令

`role-set --help` 列出了命令。下面的每一行都是当前存在的命令。标志位于每个命令的 `--help` 中。

| 命令 | 它做什么 |
|---|---|
| `plan` | 修复一个批次中从一个种子开始的每个插槽。不调用模型。相同的种子会写入相同的计划。 |
| `cost` | 在启动任何操作之前，估算计划所需的 Pod 运行时间和费用（`two-card` 或 `frontier`）。这只是一个估算。 |
| `pack sync` / `pack verify` | 从本地 aspire-si 仓库的某个提交版本复制验证器包，并将其副本与清单进行比对。 |
| `harvest` | 将从已固定的仓库中复制的、经过许可检查的摘录复制到本地池中。克隆目录不会被记录。 |
| `write` | 运行程序。`--mock` 在没有模型和没有消耗的情况下运行。实际运行需要一个环回 `--endpoint` 和一个来自 `writers.json` 的 `--model`。 |
| `readout` | 报告每个课程和类型的单次运行结果，包括通过或失败，数据来自包含 `batch-report.json` 的目录。 |
| `critic-control` | 衡量对已审查项目的评估结果。需要一个授权和一个心跳文件。`--trait-leg` 和 `--answer-leg` 运行这些环节。 |
| `ingest` / `store` | 将收据记录到存储中，防火墙位于入口处。`store status` 报告计数。`store void` 附加一个无效记录。 |
| `pilot-auto` | 付费启动器：等待环回通道，写入数据，并且可以在发生连接中断后重新启动。 |
| `puzzle-check` | 对数据文件及其 `.models.jsonl` 辅助文件运行 B 类求解器，并拒绝重复的事实骨架。 |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

生成器目前没有 `role-set` 命令。它的三个脚本生成已提交的表格和记录：

| 脚本 | 它做什么 |
|---|---|
| `scripts/t10_usable_cells.py` | 探测每个单元格并写入 `t10_cells.json`。`--check` 在已提交的表格上运行容量门控，并且不进行探测。 |
| `scripts/t10_cell_capacity.py` | 在调整种子 1、21 和 41 上，按层级统计每个单元格中不同的骨架数量，并将结果写入 `t10_capacity.json`。 |
| `scripts/t10_variety_record.py` | 将每个记录的谜题重新构建到 `t10_record.json` 中。`--check` 验证已提交的记录并打印锁定的种子数据。 |

## 门控

五种类型：

- **A.** 一段教学文本。没有结论。运行一个验证器，特征检查仅供参考。
- **B.** 一个有限的分配谜题。求解器检查模型。评估者的答案仅供参考。
- **C.** 一个带有植入缺陷的检查器，或者一个正确的检查器。结论和证据检查会拒绝。缺陷检查会被记录，但不会拒绝。
- **D.** 两个回合。结论和证据检查会拒绝。特征检查仅供参考。
- **E.** 一份书面测验，或者一份结论测验。书面 E 类型没有结论检查。结论类型的 E 检查结论和证据。

结论和证据检查会拒绝。特征、答案和缺陷检查会被记录，并且不会拒绝，直到它们的控制条件通过。特征环节的总体得分为 40.3%，而目标值为 85%；最差的特征环节得分为 12.5%，而目标值为 70%。答案环节的得分为 62.7%，而目标值为 90%。2026-10-11 的缺陷控制环节中有 9 个通过，24 个总数。

B 类求解器在其封闭的规则集合中涵盖有限的分配谜题：最多 6 个实体、3 个属性和 200,000 个分配。它不会将推理与模型进行比较。基于自由文本材料的、带有结论的 B 类项目超出了求解器的范围，并且 `puzzle-check` 将其视为超出范围，而不是通过。

存储的防火墙在摄取时拒绝保留、密封和访谈材料，并在某个阶段拒绝 DEV 材料。没有读取该材料的路径。转向功能已设计好，但从未运行。

## 一次运行会保留什么

每个接受的收据都包含计划哈希和包哈希。如果运行使用了示例，它还会包含示例提交和排除文件哈希。如果它使用了停滞规则，它会包含该规则的哈希。生成的谜题的收据包含其种子、尝试次数、读取次数、难度指标以及骨架、包和求解器哈希。每个写入调用都会被保留在 `drafts.jsonl` 中，无论是否接受。

## 模型

写入器固定为 Qwen3-235B-A22B-Instruct-2507，AWQ 构建，Apache-2.0 许可，作为 `qwen3-235b-instruct-2507` 在修订版本 `96548f657d772187f16e9f04bc50aec78a9f9cdd` 中提供。评估器固定为 `gemma4:31b`，摘要 `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`，Q4_K_M，Apache-2.0 许可，在 `http://127.0.0.1:11434` 中提供。`writers.json` 还列出了一份仅用于预演的写入器和一个用于比较的评估器。文件中不存在的名称将被拒绝，包含 `cloud` 的名称也将被拒绝。

本地 GPU 用于评估器和一个预演写入器，仅在授权内：`--grant` 和一个在 60 秒内修改的心跳文件。心跳文件不会被打开。运行加载的评估器在退出时会被卸载，包括在发生错误时。`--mock` 不加载任何内容。

## 试点

三个付费试点记录在 [docs/runs/](docs/runs/) 中，全部于 2026-10-11 进行：3.52 美元、3.61 美元和 2.80 美元。这些是本 README 中提到的唯一成本数据。没有来自这些数据的行包含在训练集中。在试点 2 中，研发部门对 49 个候选者的关键检查，有 15 个按原样通过，另外 15 个在机械修复后通过，19 个未通过。试点 3 没有写入任何内容。

## 威胁模型

role-set 是一个本地生成器。它不是服务器。它不监听，也不收集遥测数据。

一次运行会触及你命名的计划、角色包和输出目录。`ingest` 和 `store` 还会触及你命名的存储文件。`write`、`pilot-auto` 和 `critic-control` 在它们不是 `--mock` 时会触及一个环回端点。`--mock` 不发送任何内容。生成器及其脚本读取和写入仓库内的文件，并且不调用任何模型。

一次运行不会触及云模型服务、Pod、租赁帐户或租赁 API 密钥。它不会读取密封行、保留集或访谈记录。`ingest` 读取你命名的 aspire-si 仓库中的十二个公共访谈问题，在你命名的提交版本中，仅用于拒绝重复这些问题的行。它们不会被存储。`store status` 返回计数，而不是行。通过 [SECURITY.md](SECURITY.md) 中的联系方式报告漏洞。

## 测试

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

行覆盖率的最低值为 90，在 `pyproject.toml` 和 `codecov.yml` 中，低于此值运行将失败。它测量值约为 97。上面的覆盖率徽章显示了当前值。CI 还运行 atlas 检查、`role-set plan` 和 `puzzle-check` 在一个固定装置上，不使用模型。`scripts/verify` 在一个命令中运行本地检查。

手册与 [登录页面](https://mcp-tool-shop-org.github.io/role-set/) 一起发布。本 README 可以独立使用。

由 <a href="https://mcp-tool-shop.github.io/">MCP 工具商店</a> 制作。
