<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

**Protótipo, desenvolvido em código aberto.** Consultivo e experimental: as interfaces e os formatos de dados podem mudar sem aviso prévio.

O conjunto de funções define os elementos de modelagem de funções, escreve as partes que um escritor deve escrever e registra quais verificações cada elemento passou. É um gerador, não um avaliador. O que torna um bom verificador está no pacote de funções, que é escrito por humanos. Nada do que este programa escreve está em um conjunto de treinamento. Cada elemento passa pela verificação principal do departamento de P&D e por uma segunda leitura antes de se aproximar de uma.

O escritor e o crítico são acessados apenas em `http://127.0.0.1`, com uma porta explícita. Qualquer outra coisa é rejeitada antes que um byte seja enviado. O conjunto de funções nunca aluga uma máquina e nunca acessa um pod. Quando uma execução usa um pod, o "offrig" é o que o aluga, e o conjunto de funções apenas vê o túnel de loopback.

O projeto está disponível em [docs/role-set-design.md](docs/role-set-design.md) e foi concebido em consonância com o programa. As notas sobre a execução estão em [docs/runs/](docs/runs/).

## Instalação

Python 3.12, gerenciado pelo uv. Uma dependência de tempo de execução: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si), fixada em um commit do aspire-si. As verificações são importadas, não copiadas.

```text
uv sync --frozen
```

## Comandos

`role-set --help` lista os comandos. Cada linha abaixo é um comando que existe atualmente.

`plan` corrige todos os slots para um lote a partir de uma semente. Não chama um modelo. A mesma semente reproduz o plano.

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` estima as horas e os custos do pod para esse plano antes que qualquer coisa seja iniciada. O valor é uma estimativa. Os perfis são `two-card` e `frontier`.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` copia o pacote de funções de um clone local do aspire-si em um commit. `pack verify` verifica essa cópia em relação ao seu manifesto. O caminho do clone não é registrado. Um novo clone deste repositório não é um pacote sincronizado: `roles/verifier` contém a lista de exclusão e a lista de origem, e a sincronização preenche o restante.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` copia trechos com licença verificada para um pool local. O diretório do clone não é registrado.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` é o executor. `--mock` executa todo o pipeline sem modelo e sem custos, e não solicita uma concessão. A saída simulada é marcada, não é um dado e é escrita em `<out>/mock/`. Uma execução que chama um modelo precisa de `--endpoint` e `--model`, e o endpoint deve ser loopback. Carregar o crítico ou um escritor de ensaios local também precisa de `--grant` e `--heartbeat`. `--no-critic` é uma execução de teste: os elementos passam sem o segundo leitor, e o escritor do pod não usa a GPU local.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` relata uma execução por lição e tipo, incluindo uma execução que não aceitou nada. Ele lê o diretório que contém `batch-report.json`. Um `write` concluído deixa esse arquivo. `readout` é o comando que o transforma no relatório.

```text
uv run role-set readout --run out/mock
```

`critic-control` mede o crítico nos elementos revisados. Ele precisa de uma concessão e de um arquivo de sinalização. `--trait-leg` e `--answer-leg` executam essas duas medições.

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` registra os recibos no armazenamento. O firewall é executado na porta. `store status` relata o arquivo. `store void` anexa um valor nulo para uma rodada. Os leitores ignoram uma rodada com valor nulo. Não exclui linhas.

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` é o lançador pago: uma colagem, depois espera pelo canal de loopback, escreve e pode ser relançado após uma interrupção de chamada perdida. Os testes nunca o chamam em relação a um endpoint ativo. O CI usa `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` executa o solucionador do tipo B em um arquivo de dados e no "sidecar" `.models.jsonl` ao lado dele. Forneça a ele todos os arquivos de dados no diretório de uma vez: um quebra-cabeça cujo esqueleto de fatos repete um anterior, com nomes, ordem das linhas e direção deixados de lado, é rejeitado, indicando o primeiro.

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## Verificações

Cinco tipos:

- **A.** Uma passagem de ensino. Sem veredicto. Um validador é executado, e a verificação de características é consultiva.
- **B.** Um quebra-cabeça de atribuição finita. O solucionador verifica o modelo. A resposta do crítico é consultiva.
- **C.** Um verificador com uma falha plantada, ou um verificador que estava correto. As verificações de veredicto e evidência rejeitam. A verificação de falha é registrada e não rejeita.
- **D.** Duas rodadas. As verificações de veredicto e evidência rejeitam. A verificação de características é consultiva.
- **E.** Um teste escrito ou um teste de veredicto. Um E escrito não tem verificação de veredicto. Um E do tipo veredicto verifica o veredicto e a evidência.

As verificações de veredicto e evidência rejeitam. As verificações de características, resposta e falha são registradas e não rejeitam, até que seus controles sejam aprovados. A verificação de características mediu 40,3% no geral, em relação a uma barra de 85%, e a pior característica, 12,5% em relação a 70%. A verificação de resposta mediu 62,7% em relação a uma barra de 90%. O controle de falha de 11 de outubro de 2026 teve 9 de 24. Bloquear a característica ou a resposta com base nesses números rejeitaria elementos que já se sabe que são bons.

O solucionador B cobre quebra-cabeças de atribuição finita escritos em seu conjunto fechado de formas de regra: no máximo 6 entidades, 3 atributos e 200.000 atribuições. Não verifica o raciocínio em relação ao modelo. Um elemento B com chave de veredicto em material de texto livre está fora do solucionador. `puzzle-check` conta esse elemento como fora, nunca como aprovado.

O firewall do armazenamento rejeita material retido, selado e de entrevista durante a ingestão, e rejeita material DEV durante uma fase. Não há caminho de leitura para material retido, selado ou de entrevista. O armazenamento, a ingestão e o firewall são construídos. O direcionamento é projetado e nunca foi executado.

## O que uma execução mantém

Cada recibo aceito carrega o hash do plano e o hash do pacote. Quando a execução usou exemplos, o recibo também carrega o commit do exemplo e o hash do arquivo de exclusão. Quando usou a regra estagnada, o recibo carrega o hash dessa regra. Uma execução que não aceita nada não tem recibos. Cada chamada de escritor é mantida em `drafts.jsonl`, aceita ou rejeitada. Um arquivo concluído deixa `batch-report.json`, e `readout` relata esse diretório.

## Modelos

O pino do escritor é Qwen3-235B-A22B-Instruct-2507, a versão AWQ, Apache-2.0. `writers.json` o serve como `qwen3-235b-instruct-2507` na revisão `96548f657d772187f16e9f04bc50aec78a9f9cdd`.

O verificador é `gemma4:31b`, o digest é `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, o quant é Q4_K_M, Apache-2.0. O seu ponto final em `writers.json` é `http://127.0.0.1:11434`.

`writers.json` também lista um escritor que só realiza testes, `qwen3:8b`, e um verificador de comparação, `granite4.1:30b` (IBM Granite 4.1, Apache-2.0). `write` rejeita o verificador de comparação como o verificador da execução. Um nome que não está no arquivo é rejeitado. Um nome que contém `cloud` é rejeitado.

A GPU local é usada para o verificador e para um escritor de testes, e apenas dentro de um processo. Esses caminhos exigem `--grant` e um arquivo de "heartbeat" cujo tempo de modificação seja inferior a 60 segundos. `critic-control` exige ambos em cada execução. O arquivo de "heartbeat" não é aberto. Quando uma execução carrega o verificador, ele é descarregado ao sair, incluindo quando a execução é interrompida com um erro. `--mock` não carrega nada.

## Testes piloto

Três testes piloto pagos são registrados em [docs/runs/](docs/runs/). Estes são os únicos valores de custo que este arquivo README declara. Nenhum resultado do gerador deles está em um conjunto de treinamento.

- **Teste Piloto 1, 2026-10-11, $3,52.** Nenhuma das 104 linhas aceitas é um dado. A equipe de P&D manteve as 43 linhas C e D como montadas incorretamente antes de qualquer revisão de conteúdo. A nota está em [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md).
- **Teste Piloto 2, 2026-10-11, $3,61.** A verificação de chave da equipe de P&D dos 49 candidatos: 15 foram aprovados conforme escrito, mais 15 precisaram de uma correção mecânica e 19 falharam. A nota está em [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md).
- **Teste Piloto 3, 2026-10-11, $2,80.** Não escreveu nada. O motor do escritor travou após a inicialização. A nota está em [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md).

## Modelo de ameaças

O "role-set" é um gerador local. Não é um servidor. Não escuta e não coleta dados de telemetria.

Uma execução acessa o plano, o pacote de funções e o diretório de saída que você nomeia. `ingest` e `store` também acessam o arquivo de armazenamento que você nomeia. `write`, `pilot-auto` e `critic-control` acessam um ponto final de loopback quando não são `--mock`. `--mock` não envia nada.

Uma execução não acessa um serviço de modelo em nuvem, um pod, uma conta de aluguel ou uma chave de API de aluguel. Não lê linhas seladas, conjuntos retidos ou sessões de entrevista. `ingest` lê as doze perguntas públicas da entrevista do clone "aspire-si" que você nomeia, no commit que você nomeia, e apenas para que possa rejeitar uma linha que as repita. Essas perguntas não são armazenadas. `store status` retorna contagens, não linhas. Denuncie uma vulnerabilidade por meio do contato em [SECURITY.md](SECURITY.md).

## Testes

```text
uv sync --frozen
uv run pytest
```

Nesta árvore, há 241 testes e 97,64% de cobertura. O limite mínimo é 90, em `pyproject.toml` e no CI, e a execução falha abaixo desse limite. O CI também executa `role-set plan` e `puzzle-check` em um conjunto de testes, sem modelo. `scripts/verify` executa as mesmas verificações locais em um único comando.

O manual é publicado com a [página inicial](https://mcp-tool-shop-org.github.io/role-set/). Este arquivo README é independente.

Criado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
