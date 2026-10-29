<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/role-set/readme.png" alt="role-set" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/role-set/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/role-set"><img src="https://codecov.io/gh/mcp-tool-shop-org/role-set/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/role-set/"><img src="https://img.shields.io/pypi/v/role-set" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/role-set/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**原型，以开放方式开发。** 仅供参考和实验：接口和数据结构会不经通知地发生变化。在有人检查之前，角色集编写的任何内容都不会被视为训练数据。每个项目都会经过研发部门的关键检查和第二次审查，然后才能接近训练集。

角色集规划角色塑造项目，编写作者应该编写的部分，并记录每个项目通过了哪些关卡。它还在代码中构建 B 类谜题，并在模型看到它们之前对其进行验证。它是一个生成器，而不是一个裁判。一个好的角色持有者存在于角色包中，而角色包是由人们编写的。

作者和评论者只能通过 `http://127.0.0.1` 显式端口进行访问。任何其他访问都将被拒绝，并且在发送任何字节之前。角色集绝不会租用机器，也不会触碰容器。当一个运行使用容器时，offrig 会租用它，而角色集只会看到环回隧道。

设计文档为 [docs/role-set-design.md](docs/role-set-design.md) 和 [docs/generator-design.md](docs/generator-design.md)，与程序保持一致。运行日志位于 [docs/runs/](docs/runs/)。

## 安装

Python 3.13 或更高版本（3.15 是主要目标；CI 测试 3.13、3.14 和 3.15）。一个运行时依赖项：[aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 1.7.6，已固定版本。这些关卡是导入的，而不是复制的。

```text
pip install role-set
```

要使用角色集本身，请克隆它，并让 uv 安装锁定的环境：

```text
uv sync --frozen
```

## 角色包

角色包是人们为某个角色编写的文本。角色集以切片的形式提供，并且不会对其进行任何修改。它的哈希值位于每个已接受的收据中。

- **验证器** (`roles/verifier`)。`pack sync` 从本地 aspire-si 克隆的特定提交版本中复制：包括课程、18 个特征、类型表、修订和规则。`pack verify` 将副本与其清单进行比较。新的克隆只包含源列表、排除列表和 `defects.jsonl`，即审查发现的缺陷类别目录。同步会填充其余部分。
- **倡导者** (`roles/advocate`)。一个人编写的包，其中不包含任何模型文本。总监于 2026-10-14 批准了该包（在拉取请求 #56 上的提交 `ffb9546`），并且第一个批次正在构建；它不会在 `main` 中显示，直到 #56 合并。第一个批次包含三个特征：`failed-gate-first`、`exact-figures` 和 `faithful-relay`。在通过试点阅读之前，任何倡导者项目都不会被视为训练数据。

## T10 生成器

B 类谜题过去是手工编写，然后进行检查。现在，代码构建谜题并对其进行验证，然后模型仅对其进行描述。该生成器是 `role_set.generate` 包。T10，关于严格大小词（“完全”、“至少”、“最多”）的课程，是 0.2.0 版本中发生更改的部分。还会构建一个 T12 方案（在特定日期生效的版本）。T11 不会构建。

**通过构建进行验证。** 构建器会采样一个隐藏的解决方案，添加在解决方案中为真的线索，直到答案确定，然后删除答案不需要的每个线索。求解器检查模型。然后，构建器运行研发部门的四个 T10 阅读：将“完全”读作“至少”，“完全”读作“最多”，“至少”读作“完全”，以及“最多”读作“完全”。只有当这些阅读中的一个应用于特征线索时，才会保留谜题，并且该阅读会改变答案。如果谜题在其他计数线索上通过，则该谜题是缺陷类别 `trait_not_tested`，并且会被拒绝。

**单元格映射。** 每个包、问题和种子都映射到结构单元格：实体计数 `n`（3 到 5），计数词 `op`，`k`（1 到 n-1），以及 `ifs`，即 `if` 线索的数量（0 到 2）。设置问题使用所有三个计数词。声明问题仅使用“完全”，因为“至少”和“最多”无法改变声明的结论。可用的单元格会进行一次探测，并在 `t10_cells.json` 中提交。

**声明轴 `if_shape`。** 声明单元格还命名其 `if` 线索的形状。有七种形状：`none` 表示没有 `if`；`pos` 或 `neg` 表示一个 `if`，由其结论的极性决定；以及 `chain`、`shared_consequent`、`shared_antecedent` 或 `disjoint` 表示两个。如果构建的简化谜题失去了其形状，则它会在同一个种子中重试。

**分配器。** 对于一个问题和一个值计数，一个包含 20 个种子的文件会在任何构建开始之前分配一次，通过最小成本流进行分配。每个包的 20 个种子使用 20 个不同的单元格。一个没有 `if` 线索的单元格在一个文件中最多使用两次（零如果限制）。其他访问优先使用在调整种子中产生更多不同骨架的单元格。这些权重分别针对两个值和三个值的包进行保存。

**多样性下限。** 在每个层级（设置和声明，两个和三个值）：至少 75% 的不同规范骨架，没有骨架出现超过 3 次，并且在一个包的文件中没有骨架出现两次。这些下限是在锁定的种子上进行判断的。在锁定的种子 101 到 120 上，三个值的声明达到了 118/120 个不同的骨架，并且最高的重复次数为 2。

**已提交的记录。** `t10_record.json` 包含 1,280 个构建（种子 1-20、61-80、81-100 和 101-120，每个包，两个问题），每个构建都有其单元格、`if_shape`、尝试次数和规范骨架。它携带一个 sha256，用于验证生成器源代码、单元格表、容量表和包数据。一个测试会确保该文件列表在角色集自己的导入下是封闭的。正常的 CI 测试作业会在几分钟内检查记录中的哈希值和下限。对每个谜题进行详尽的重建，作为 `sweep` 作业运行，仅在 `workflow_dispatch` 上运行，并且如果与记录有任何差异，则会失败。

**容量门和剩余空间。** `scripts/t10_usable_cells.py --check` 报告每个层级的供应量、需求量和剩余空间，并且在无法放置文件时会失败。通过 `if_shape`，三个值的包之间总共有 61 个可用的声明单元格，而在之前只有 27 个。没有包需要进行零 `if` 访问，并且一个测试会放置十二个三个值的包（六个真实的包和六个克隆包），并且在一个文件中没有重复。

## 命令

`role-set --help` 列出了命令。下面的每一行都是当前存在的命令。标志位于每个命令的 `--help` 中。

| 命令 | 它的作用 |
|---|---|
| `plan` | 修复了从种子开始，一批数据的每个插槽。没有模型调用。相同的种子会生成相同的计划。 |
| `cost` | 在启动任何内容之前（`two-card` 或 `frontier`），估算一个计划所需的 Pod 运行时间和成本。这只是一个估算。 |
| `pack sync` / `pack verify` | 从本地的 aspire-si 克隆的特定提交版本复制验证器包，并将其副本与清单进行检查。 |
| `harvest` | 将从已固定仓库中复制的、经过许可检查的摘录复制到本地池中。克隆目录不会被记录。 |
| `write` | 运行程序。`--mock` 在没有模型和没有消耗的情况下运行它。实际运行需要一个环回 `--endpoint` 和一个来自 `writers.json` 的 `--model`。`--wave write`、`critique` 和 `rewrite` 将一批数据分成几部分，以便编写器和评论者可以依次运行，如果需要，可以在不同的主机上运行，并在它们之间使用 `wave-drafts.jsonl`。 |
| `readout` | 报告每个课程和类型的单次运行结果，包括通过或失败，数据来自包含 `batch-report.json` 的目录。 |
| `critic-control` | 在 llama-server 上衡量评论器对已审核项目的表现。需要评论器的 GGUF 和服务器哈希，以及机器信息，此外还需要来自租赁主机的授权和心跳文件。`--trait-leg` 和 `--answer-leg` 运行这些步骤。 |
| `ingest` / `store` | 将收据记录到存储中，防火墙位于入口处。`store status` 报告计数。`store void` 附加一个无效记录。 |
| `pilot-auto` | 2026-10-11 试点项目的付费启动器。其脚本命名了试点项目的 Qwen 和 Gemma 固定点，这些固定点不在 `writers.json` 中，因此 `write` 会拒绝它。 |
| `puzzle-check` | 对数据文件及其 `.models.jsonl` 辅助文件运行 B 类求解器，并拒绝重复的事实骨架。 |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

生成器目前没有 `role-set` 命令。它的三个脚本创建已提交的表格和记录：

| 脚本 | 它的作用 |
|---|---|
| `scripts/t10_usable_cells.py` | 探测每个单元格并写入 `t10_cells.json`。`--check` 在已提交的表格上运行容量门控，并且不进行探测。 |
| `scripts/t10_cell_capacity.py` | 在调整的种子 1、21 和 41 上，按层级统计每个单元格中不同的骨架数量，并将结果存储到 `t10_capacity.json` 中。 |
| `scripts/t10_variety_record.py` | 将每个记录的谜题重新构建到 `t10_record.json` 中。`--check` 验证已提交的记录并打印锁定种子的数据。 |

## 门控

五种类型：

- **A.** 一段教学文本。没有结论。运行一个验证器，特征检查仅供参考。
- **B.** 一个有限的分配谜题。求解器检查模型。评论者的答案仅供参考。
- **C.** 一个带有植入缺陷的检查器，或者一个正确的检查器。结论和证据检查会拒绝。缺陷检查会被记录，但不会拒绝。
- **D.** 两个回合。结论和证据检查会拒绝。特征检查仅供参考。
- **E.** 一份书面测验，或一份结论测验。书面 E 类型没有结论检查。结论类型的 E 检查结论和证据。

结论和证据检查会拒绝。特征、答案和缺陷检查会被记录，并且不会拒绝，直到它们的控制项通过。特征测试的总体结果为 40.3%，而目标值为 85%；最差的特征测试结果为 12.5%，而目标值为 70%。答案测试结果为 62.7%，而目标值为 90%。2026-10-11 缺陷控制项为 9/24。

B 类求解器在其封闭的规则形式集合中涵盖有限的分配谜题：最多 6 个实体、3 个属性和 200,000 个分配。它不会将推理与模型进行检查。基于自由文本材料的、带有结论的 B 类项目超出了求解器的范围，并且 `puzzle-check` 将其视为超出范围，而不是通过。

存储的防火墙在摄取时拒绝保留、密封和访谈材料，并在某个阶段拒绝 DEV 材料。没有读取该材料的路径。转向功能已设计好，但从未运行。

## 运行会保留哪些内容

每个接受的收据都包含计划哈希和包哈希。如果运行使用了示例，它还会包含示例提交和排除文件哈希。如果它使用了受限规则，它会包含该规则的哈希。生成的谜题的收据包含其种子、尝试次数、读取次数、难度指标以及骨架、包和求解器哈希。每个编写器调用都会被保存在 `drafts.jsonl` 中，无论是否被接受或拒绝。

## 模型

`writers.json` 是编写器列表。除非制造商的许可或条款中的某个条款限制了使用，否则允许使用；沉默不构成限制。每个条目都会记录条款搜索：页面的 URL、读取日期、读取者、其在 rnd 许可注册中的 `registry_id`（`licences/registry.json`）以及其训练结论（`allowed`、`conditions` 或 `barred`）。如果缺少该记录，或者其结论为 `barred`，则在加载列表时会拒绝该条目。列表中不存在的名称会被拒绝，并且包含 `cloud` 的名称也会被拒绝。

编写器包括 Nemotron 3 Ultra 550B、DeepSeek-R1-0528、Nemotron 3.5 Super 120B（这三个都在一个 Pod 中），以及试点项目的八个分支：Nemotron 3.5 Bolt 74B（NVIDIA 自带的 24GB 感知量化构建）、Muse-Glimmer-30B、Nemotron 3.5 Lightning 30B、Qwen3.8-27B、Qwen3.6-35B-A3B、Granite-4.2-30B、Gemma-4-31B-it 和 DeepSeek-R1-Distill-Qwen-32B。评论器是 Muse-Glimmer-30B 和 Nemotron 3.5 Lightning。拒绝使用编写器自身系列的评论器。每个已固定的条目都包含一个 `pin_note`：其 GGUF 的制作方式，以及它是否适合试点项目中的 32,768 个令牌上下文的 40 GB A100。

模型由 llama.cpp 的 `llama-server` 提供服务，使用 CUDA 13.4 构建，位于 Q8_0。在模型被固定之前，会拒绝已列出的模型：其仓库修订版本、GGUF 文件以及文件的 sha256。Wave 脚本在启动之前对 GGUF 和 `llama-server` 二进制文件进行哈希处理，并且运行会拒绝与列表的任何偏差。每个收据都会记录分区（`production` 或 `trial:<name>`）、两个模型的系列、修订版本和 GGUF 哈希、服务器的提交和哈希，以及 CUDA 机器信息。

**卡片值。** 编写者和评论者都使用其制造商记录的设置运行，这些设置从 rnd 的 `experiments/local-builders/card-values.json` 读取，并使用 `--cards`（文件 J2 读取，因此模型在两者中以相同的方式运行）。每个采样密钥和聊天模板设置都会在每次调用中应用，包括禁用的值，因此 `llama-server` 自身的默认值永远不会传递到模型。服务器以 32,768 个令牌的上下文运行，并且回复可以使用 16,384 个令牌，因为每个具有思考模式的模型都会进行思考。R1-Distill 的系统文本会折叠到其用户回合中，如其卡片所要求。编写者通过其在 `writers.json` 中的 `card` ID 或其服务名称与卡片匹配。收据会记录文件的 sha256、候选对象以及发送的确切内容，用于编写者和评论者。如果缺少 `--cards`，则拒绝真实的 `write` 或 `critic-control`。

一个步骤仅在授权范围内运行，除非其机器行指定了 Colab VM 或 pod：`--grant` 和一个心跳文件，该文件在 60 秒前修改。无论其卡片上显示的内容如何，都需要一条指定设备的行。不会打开心跳文件。`--mock` 不加载任何内容。

## 试点

三个付费试点项目记录在 [docs/runs/](docs/runs/) 中，全部于 2026-10-11 进行：3.52 美元、3.61 美元和 2.80 美元。这些是本 README 中提到的唯一成本数据。其中没有一行数据包含在训练集中。在试点 2 中，研发部门对 49 个候选对象进行了关键检查，其中 15 个按原样通过，另外 15 个在进行机械修复后通过，19 个未通过。试点 3 没有进行任何操作。

## 威胁模型

role-set 是一个本地生成器。它不是服务器。它不监听，也不收集遥测数据。

一个运行会访问您指定的计划、角色包和输出目录。`ingest` 和 `store` 还会访问您指定的存储文件。`write`、`pilot-auto` 和 `critic-control` 在它们不是 `--mock` 时会访问一个环回端点。`--mock` 不发送任何内容。生成器及其脚本读取和写入仓库内的文件，并且不调用任何模型。

一个运行不会访问云模型服务、pod、租赁帐户或租赁 API 密钥。它不会读取密封行、保留集或访谈记录。`ingest` 从您指定的 aspire-si 克隆中读取十二个公开的访谈问题，并指定您使用的提交版本，仅用于拒绝重复这些问题的行。这些问题不会被存储。`store status` 返回计数，而不是行。请通过 [SECURITY.md](SECURITY.md) 中的联系方式报告漏洞。

## 测试

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

行覆盖率的最低值为 90，在 `pyproject.toml` 和 `codecov.yml` 中，如果低于此值，则运行将失败。它大约为 97。上面的覆盖率徽章显示了当前值。CI 还会运行 atlas 检查、`role-set plan` 和 `puzzle-check`，在测试环境中运行，不使用任何模型。`scripts/verify` 在一个命令中运行本地检查。

## 贡献者

本仓库的部分内容使用 Grok (xAI) 编写。Grok、Argon 和 Claude 在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 中获得了认可。

该手册与 [登录页面](https://mcp-tool-shop-org.github.io/role-set/) 一起发布。本 README 可以独立使用。

由 <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> 构建
