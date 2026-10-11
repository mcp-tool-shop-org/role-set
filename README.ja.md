<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**オープンな環境で開発されたプロトタイプ。** 助言的かつ実験的：インターフェースとデータ構造は予告なく変更される。

role-setは、役割を定義する計画、ライターが書くべき部分、および各項目が通過したゲートを記録する。これはジェネレーターであり、審査者ではない。優れた検証者の基準は、人間が作成する役割パックに存在する。このプログラムが生成する内容は、トレーニングセットに含まれない。すべての項目は、R&Dによる主要チェックと、最終的な評価の前に2回目の確認を受ける。

ライターと批評家は、明示的なポートを介してのみ`http://127.0.0.1`にアクセスできる。それ以外は、1バイトのデータ送信前に拒否される。role-setは、マシンをレンタルしたり、ポッドにアクセスしたりしない。実行時にポッドを使用する場合、offrigがレンタルを行い、role-setはループバックトンネルのみを認識する。

デザインは[docs/role-set-design.md]に記載されており、プログラムと整合性が保たれています。実行に関する注記は[docs/runs/]にあります。

## インストール

Python 3.12、uvによって管理。1つの実行時依存関係：[aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si)、aspire-siのコミットに固定。ゲートはインポートされ、コピーされない。

```text
uv sync --frozen
```

## コマンド

`role-set --help`は、コマンドを一覧表示する。以下に、現在存在するコマンドを1つずつ示す。

`plan`は、シードからバッチのすべてのスロットを修正する。モデルは呼び出さない。同じシードを使用すると、同じ計画が再現される。

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost`は、実行前に、その計画に必要なポッド時間とコストを見積もる。この数値はあくまで見積もりである。プロファイルは`two-card`と`frontier`である。

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync`は、ローカルのaspire-siクローンから役割パックを、特定のコミットでコピーする。`pack verify`は、そのコピーをマニフェストと照合して検証する。クローンのパスは記録されない。このリポジトリの新しいクローンは、同期されたパックではない。`roles/verifier`には、除外リストとソースリストが含まれており、同期によって残りの部分が補完される。

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest`は、ライセンスチェック済みの抜粋をローカルプールにコピーする。クローンディレクトリは記録されない。

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write`は、ランナーである。`--mock`は、モデルやコストを使用せずに、パイプライン全体を実行し、許可を要求しない。モック出力はマークされ、データではなく、`<out>/mock/`の下に書き込まれる。モデルを呼び出す実行には、`--endpoint`と`--model`が必要であり、エンドポイントはループバックでなければならない。批評家、またはローカルのリハーサルライターをロードするには、`--grant`と`--heartbeat`も必要である。`--no-critic`は、テスト実行である。項目は2回目のレビューなしで通過し、ポッドライターはローカルGPUを使用しない。

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout`は、レッスンと種類ごとに1つの実行をレポートする。これには、何も受け入れなかった実行も含まれる。`batch-report.json`が含まれるディレクトリを読み取る。完了した`write`は、そのファイルを残す。`readout`は、それをレポートに変換するコマンドである。

```text
uv run role-set readout --run out/mock
```

`critic-control`は、レビューされた項目に基づいて批評家を評価する。許可とハートビートファイルが必要である。`--trait-leg`と`--answer-leg`は、これらの2つの評価を実行する。

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest`は、受領書をストアに記録する。ファイアウォールは、入口で実行される。`store status`は、ファイルをレポートする。`store void`は、ラウンドの無効化を追加する。レビュー担当者は、無効化されたラウンドをスキップする。行は削除されない。

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto`は、有料のランチャーである。1回のペーストを行い、ループバックレーンを待機し、書き込みを行い、中断された場合に再起動できる。テストでは、ライブエンドポイントに対してこれを呼び出すことはない。CIでは、`write --mock`を使用する。

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check`は、種類Bのソルバーをデータファイルと、その隣に配置された`.models.jsonl`のサイドカーに対して実行する。ディレクトリ内のすべてのデータファイルを一度に与える。事実の骨格が、名前、行の順序、および方向を無視して、以前のものを繰り返すパズルは拒否され、最初のものが示される。

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## ゲート

5つの種類：

- **A.** 指導的な文章。評価は行われない。検証者が実行され、特性チェックは助言的なものである。
- **B.** 有限な割り当てパズル。ソルバーがモデルをチェックする。批評家の回答は助言的なものである。
- **C.** 意図的に欠陥が埋め込まれたチェッカー、または正しかったチェッカー。評価と証拠のチェックは拒否される。欠陥チェックは記録され、拒否は行われない。
- **D.** 2つのターン。評価と証拠のチェックは拒否される。特性チェックは助言的なものである。
- **E.** 書面によるクイズ、または評価クイズ。書面によるEには、評価チェックは含まれない。評価型のEは、評価と証拠をチェックする。

評価と証拠のチェックは拒否される。特性、回答、および欠陥のチェックは記録され、それらの制御が合格するまで拒否は行われない。特性の評価は、全体で85%の基準に対して40.3%であり、最悪の特性は70%の基準に対して12.5%であった。回答の評価は、90%の基準に対して62.7%であった。2026年10月11日の欠陥制御では、24のうち9であった。これらの数値に基づいて特性または回答をゲートすると、すでに良好であることがわかっている項目が拒否されることになる。

Bソルバーは、その閉じたルール形式で記述された有限な割り当てパズルを扱う。最大6つのエンティティ、3つの属性、および20万の割り当て。思考をモデルに対してチェックすることはない。自由テキストの材料に対する評価キーのB項目は、ソルバーの範囲外である。`puzzle-check`は、その項目を範囲外としてカウントし、合格とはカウントしない。

ストアのファイアウォールは、取り置き、封印、およびインタビューの材料をインジェスト時に拒否し、ステージ中にDEV材料を拒否する。取り置き、封印、またはインタビューの材料に対する読み取りパスはない。ストア、インジェスト、およびファイアウォールは構築されている。ステアリングは設計されており、まだ実行されていない。

## 実行が保持するもの

すべての受け入れられた受領書には、計画ハッシュとパックハッシュが含まれる。実行で例を使用した場合、受領書には、例のコミットと除外ファイルハッシュも含まれる。固定ルールを使用した場合、受領書には、そのルールのハッシュが含まれる。何も受け入れなかった実行では、受領書は作成されない。すべてのライターの呼び出しは、`drafts.jsonl`に保持され、受け入れまたは拒否される。完了した書き込みは、`batch-report.json`を残し、`readout`がそのディレクトリをレポートする。

## モデル

ライターピンはQwen3-235B-A22B-Instruct-2507、AWQビルド、Apache-2.0である。`writers.json`は、これをリビジョン`96548f657d772187f16e9f04bc50aec78a9f9cdd`の`qwen3-235b-instruct-2507`として提供する。

批判モデルは`gemma4:31b`、要約モデルは`6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`、量子化はQ4_K_M、ライセンスはApache-2.0です。`writers.json`におけるエンドポイントは`http://127.0.0.1:11434`です。

`writers.json`には、リハーサル専用のライター`qwen3:8b`と、比較批判モデル`granite4.1:30b`（IBM Granite 4.1、Apache-2.0）もリストされています。`write`は、実行時の批判モデルとして比較批判モデルを拒否します。ファイルに存在しない名前は拒否されます。`cloud`を含む名前は拒否されます。

ローカルGPUは、批判モデルとリハーサルライターに使用され、許可された範囲内でのみ使用されます。これらのパスには、`--grant`と、最終更新時刻が60秒未満のハートビートファイルが必要です。`critic-control`は、すべての実行時にこれらを必要とします。ハートビートファイルは開かれません。実行時に批判モデルがロードされた場合、エラーで実行が停止した場合を含む、終了時にアンロードされます。`--mock`は何もロードしません。

## パイロットテスト

3つの有料パイロットテストが[docs/runs/](docs/runs/)に記録されています。これらは、このREADMEに記載されている唯一のコスト情報です。これらのテストから生成された出力は、トレーニングセットに含まれていません。

- **パイロットテスト1、2026-10-11、$3.52。** 104行のうち、データとして受け入れられた行はありません。R&Dは、コンテンツレビューの前に、43行のCおよびDを誤って組み立てられたものとして扱いました。注記は[docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md)です。
- **パイロットテスト2、2026-10-11、$3.61。** R&Dによる49件の候補のキーチェック：15件は記述通りに合格、さらに15件は機械的な修正が必要、19件は不合格でした。注記は[docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md)です。
- **パイロットテスト3、2026-10-11、$2.80。** 何も書き込みませんでした。ライターのエンジンは、起動後に停止しました。注記は[docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md)です。

## 脅威モデル

role-setはローカルジェネレーターです。サーバーではありません。リッスンも、テレメトリーの収集も行いません。

実行は、プラン、ロールパック、および指定した出力ディレクトリにアクセスします。`ingest`と`store`は、指定したストアファイルにもアクセスします。`write`、`pilot-auto`、および`critic-control`は、それらが`--mock`でない場合に、ループバックエンドポイントにアクセスします。`--mock`は何も送信しません。

実行は、クラウドモデルサービス、ポッド、レンタルアカウント、またはレンタルAPIキーにはアクセスしません。シールされた行、保留されたセット、またはインタビューセッションも読み取りません。`ingest`は、指定したコミットにおける、指定したaspire-siクローンから、12個の公開インタビューの質問を読み取ります。これは、質問を繰り返す行を拒否するためだけです。これらの質問は保存されません。`store status`は、行ではなく、カウントを返します。脆弱性を発見した場合は、[SECURITY.md](SECURITY.md)の連絡先に報告してください。

## テスト

```text
uv sync --frozen
uv run pytest
```

Line coverage has a floor of 90, in `pyproject.toml` and in CI, and the run fails under it. The coverage badge above shows the current figure. CI also runs `role-set plan` and `puzzle-check` on a fixture, with no model. `scripts/verify` runs the same local checks in one command.

ハンドブックは、[ランディングページ](https://mcp-tool-shop-org.github.io/role-set/)で公開されています。このREADMEは、それ自体で完結しています。

MCP Tool Shopによって作成されました。
