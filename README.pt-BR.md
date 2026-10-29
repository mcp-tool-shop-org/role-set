<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.md">English</a>
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

**Protótipo, desenvolvido em código aberto.** Consultivo e experimental: as interfaces e os formatos de dados são alterados sem aviso prévio. Nada do que é definido em um conjunto de papéis é considerado dado de treinamento até que seja verificado por uma pessoa. Cada item passa pela verificação principal da equipe de P&D e por uma segunda análise antes de se aproximar de um conjunto de treinamento.

O plano do conjunto de papéis define os itens que moldam o papel, escreve as partes que um escritor deve escrever e registra quais critérios cada item passou. Ele também cria quebra-cabeças do tipo B em código e os valida antes que um modelo os veja. É um gerador, não um avaliador. O que define um bom conjunto de papéis está em um pacote de papéis, que as pessoas escrevem.

O escritor e o crítico são acessados apenas em `http://127.0.0.1`, por meio de uma porta explícita. Qualquer outra tentativa é recusada antes que um único byte seja enviado. O conjunto de papéis nunca aluga uma máquina e nunca acessa um pod. Quando uma execução usa um pod, a offrig o aluga, e o conjunto de papéis só vê o túnel de loopback.

Os projetos estão em [docs/role-set-design.md](docs/role-set-design.md) e [docs/generator-design.md](docs/generator-design.md), mantidos em sincronia com o programa. As notas da execução estão em [docs/runs/](docs/runs/).

## Instalação

Python 3.13 ou mais recente (3.15 é o objetivo principal; os testes de CI usam 3.13, 3.14 e 3.15). Uma única dependência de tempo de execução: [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 1.7.6, fixada. Os critérios são importados, não copiados.

```text
pip install role-set
```

Para trabalhar no próprio conjunto de papéis, clone-o e deixe que o uv instale o ambiente bloqueado:

```text
uv sync --frozen
```

## Os pacotes de papéis

Um pacote de papéis é o texto que as pessoas escreveram para um determinado papel. O conjunto de papéis o fornece em partes e não adiciona nada a ele. Seu hash está em cada recibo aceito.

- **O verificador** (`roles/verifier`). `pack sync` copia-o de um clone local do aspire-si em um commit: as lições, os 18 traços, a tabela de tipos, as alterações e as regras. `pack verify` verifica a cópia em relação ao seu manifesto. Um clone novo contém apenas a lista de fontes, a lista de exclusões e `defects.jsonl`, o catálogo de classes de defeitos que as revisões encontraram. A sincronização preenche o restante.
- **O defensor** (`roles/advocate`). Um pacote que uma pessoa escreveu neste repositório, sem nenhum texto de modelo. O diretor aprovou o pacote em 14 de outubro de 2026 (commit `ffb9546` no pull request #56), e o lote 1 está sendo criado lá; ele não está em `main` até que o #56 seja mesclado. O lote 1 tem três traços: `failed-gate-first`, `exact-figures` e `faithful-relay`. Nenhum item do defensor é considerado dado de treinamento até que passe pela análise piloto.

## O gerador T10

Os quebra-cabeças do tipo B costumavam ser escritos à mão e, em seguida, verificados. Agora, o código cria o quebra-cabeça e o valida, e um modelo apenas o formula. O gerador é o pacote `role_set.generate`. T10, a lição sobre palavras de tamanho estrito ("exatamente", "pelo menos", "no máximo"), é a parte que mudou na versão 0.2.0. Uma receita T12 (qual edição estava em vigor em uma determinada data) também é criada. T11 não é.

**Validado pela construção.** O criador amostra uma solução oculta, adiciona pistas que são verdadeiras nela até que a resposta seja definida e exclui todas as pistas de que a resposta não precisa. O solucionador verifica o modelo. Em seguida, o criador executa as quatro análises T10 da equipe de P&D: "exatamente" lido como "pelo menos", "exatamente" lido como "no máximo", "pelo menos" lido como "exatamente" e "no máximo" lido como "exatamente". Um quebra-cabeça é mantido apenas quando uma dessas análises, aplicada à pista do traço sozinha, altera a resposta. Um quebra-cabeça que passa em alguma outra contagem de pistas é a classe de defeito `trait_not_tested`, e é recusado.

**O mapa de células.** Cada pacote, pergunta e semente é mapeado para uma célula estrutural: a contagem de entidades `n` (3 a 5), a palavra de contagem `op`, `k` (1 a n−1) e `ifs`, o número de pistas `if` (0 a 2). As perguntas de conjunto usam as três palavras de contagem. As perguntas de afirmação usam apenas "exatamente", porque "pelo menos" e "no máximo" não podem alterar o veredicto de uma afirmação. As células utilizáveis são pesquisadas uma vez e confirmadas em `t10_cells.json`.

**O eixo de afirmação `if_shape`.** Uma célula de afirmação também nomeia a forma de suas `if` pistas. Existem sete formas: `none` para nenhuma `if`; `pos` ou `neg` para uma `if`, pela polaridade de sua consequência; e `chain`, `shared_consequent`, `shared_antecedent` ou `disjoint` para duas. Uma construção cujo quebra-cabeça reduzido perde sua forma tenta novamente dentro da mesma semente.

**O alocador.** Para uma pergunta e uma contagem de valores, um arquivo de 20 sementes é atribuído uma vez, antes que qualquer coisa seja criada, por meio de um fluxo de custo mínimo. As 20 sementes de cada pacote ocupam 20 células diferentes. Uma célula sem uma pista `if` é usada no máximo duas vezes em um arquivo (o limite de zero, se houver). As visitas subsequentes favorecem as células que forneceram mais esqueletos distintos nas sementes de ajuste. Esses pesos são mantidos separados para pacotes de dois e três valores.

**Os limites de variedade.** Em cada estrato (conjunto e afirmação, dois e três valores): pelo menos 75% de esqueletos canônicos distintos, nenhum esqueleto mais de 3 vezes e nenhum esqueleto duas vezes no mesmo arquivo de um pacote. Os limites são avaliados em sementes bloqueadas. Nas sementes bloqueadas de 101 a 120, as afirmações de três valores atingem 118/120 distintos, com um máximo de 2 repetições.

**O registro confirmado.** `t10_record.json` contém 1.280 construções (sementes 1–20, 61–80, 81–100 e 101–120, cada pacote, ambas as perguntas), cada uma com sua célula, `if_shape`, tentativas e esqueleto canônico. Ele carrega um sha256 sobre a fonte do gerador, a tabela de células, a tabela de capacidade e os dados do pacote. Um teste mantém essa lista de arquivos fechada sob as próprias importações do conjunto de papéis. O trabalho normal de teste de CI verifica o hash e os limites do registro em minutos. A reconstrução exaustiva de cada quebra-cabeça é executada como o trabalho `sweep`, apenas em `workflow_dispatch`, e falha em qualquer diferença em relação ao registro.

**O critério de capacidade e a margem.** `scripts/t10_usable_cells.py --check` relata a oferta, a demanda e a margem para cada estrato e falha com os números quando um arquivo não pode ser colocado. Com `if_shape`, os pacotes de três valores têm 61 células de afirmação utilizáveis entre eles, em comparação com 27 antes. Nenhum pacote precisa de uma visita de zero-`if`, e um teste coloca doze pacotes de três valores (os seis reais e seis clones) sem repetição em um arquivo.

## Comandos

`role-set --help` lista os comandos. Cada linha abaixo é um comando que existe hoje. As flags estão em cada comando `--help`.

| Comando | O que ele faz |
|---|---|
| `plan` | Corrige todos os espaços para um lote a partir de uma semente. Sem chamada de modelo. A mesma semente gera o mesmo plano. |
| `cost` | Estima as horas e os custos de um plano antes de qualquer lançamento (`two-card` ou `frontier`). É uma estimativa. |
| `pack sync` / `pack verify` | Copia o pacote de verificação de um clone local do aspire-si em um commit específico e verifica a cópia em relação ao seu manifesto. |
| `harvest` | Copia trechos com licença verificada de repositórios fixados para um pool local. O diretório do clone não é registrado. |
| `write` | O executor. `--mock` executa sem modelo e sem custo. Uma execução real precisa de um loopback `--endpoint` e um `--model` de `writers.json`. `--wave write`, `critique` e `rewrite` dividem um lote para que o escritor e o crítico executem um de cada vez, em hosts diferentes, se necessário, com `wave-drafts.jsonl` entre eles. |
| `readout` | Relata uma execução por lição e tipo, aprovada ou reprovada, a partir do diretório que contém `batch-report.json`. |
| `critic-control` | Avalia um crítico em itens revisados, em seu servidor llama. Precisa dos hashes GGUF e do servidor do crítico, bem como da linha da máquina, além de uma permissão e um arquivo de sinalização de um host alugado. `--trait-leg` e `--answer-leg` executam essas etapas. |
| `ingest` / `store` | Registra os recibos no armazenamento, com o firewall na entrada. `store status` relata as contagens. `store void` adiciona um recibo de anulação. |
| `pilot-auto` | O lançador pago dos pilotos de 2026-10-11. Seu script nomeia os repositórios Qwen e Gemma dos pilotos, que não estão em `writers.json`, então `write` o rejeita. |
| `puzzle-check` | Executa o solucionador do tipo B em arquivos de dados e seus `.models.jsonl` arquivos auxiliares, e rejeita um esqueleto de fatos repetido. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

O gerador ainda não tem o comando `role-set`. Seus três scripts criam as tabelas confirmadas e o registro:

| Script | O que ele faz |
|---|---|
| `scripts/t10_usable_cells.py` | Verifica cada célula e grava `t10_cells.json`. `--check` executa o controle de capacidade na tabela confirmada e não verifica. |
| `scripts/t10_cell_capacity.py` | Conta os esqueletos distintos por célula nas sementes de ajuste 1, 21 e 41, por estrato, em `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Reconstrói cada quebra-cabeça registrado em `t10_record.json`. `--check` verifica o registro confirmado e imprime os valores da semente bloqueada. |

## Controles

Cinco tipos:

- **A.** Um trecho de ensino. Sem veredicto. Um validador é executado, e a verificação de características é apenas uma recomendação.
- **B.** Um quebra-cabeça de atribuição finita. O solucionador verifica o modelo. A resposta do crítico é apenas uma recomendação.
- **C.** Um verificador com uma falha intencional ou um verificador que estava correto. As verificações de veredicto e evidência são rejeitadas. A verificação de falha é registrada e não rejeita.
- **D.** Duas rodadas. As verificações de veredicto e evidência são rejeitadas. A verificação de características é apenas uma recomendação.
- **E.** Um teste escrito ou um teste de veredicto. Um teste escrito do tipo E não tem verificação de veredicto. Um teste do tipo E com veredicto verifica o veredicto e a evidência.

As verificações de veredicto e evidência são rejeitadas. As verificações de características, resposta e falha são registradas e não rejeitam, até que seus controles sejam aprovados. O teste de características mediu 40,3% no geral, em comparação com uma barra de 85%, e o pior teste de característica, 12,5%, em comparação com 70%. O teste de resposta mediu 62,7% em comparação com uma barra de 90%. O controle de falha de 2026-10-11 tinha 9/24.

O solucionador B cobre quebra-cabeças de atribuição finita em seu conjunto fechado de formas de regra: no máximo 6 entidades, 3 atributos e 200.000 atribuições. Ele não verifica o raciocínio em relação ao modelo. Um item do tipo B com veredicto sobre material de texto livre está fora do solucionador, e `puzzle-check` o conta como externo, nunca como aprovado.

O firewall do armazenamento rejeita material retido, selado e de entrevista durante a ingestão, e material de DESENVOLVIMENTO durante uma etapa. Não há caminho de leitura para esse material. O direcionamento é projetado e nunca foi executado.

## O que uma execução mantém

Cada recibo aceito contém o hash do plano e o hash do pacote. Quando a execução usou exemplos, também contém o commit do exemplo e o hash do arquivo de exclusão. Quando usou a regra estagnada, contém o hash dessa regra. O recibo de um quebra-cabeça gerado contém sua semente, tentativas, leituras, métricas de dificuldade e os hashes do esqueleto, pacote e solucionador. Cada chamada do escritor é mantida em `drafts.jsonl`, aceita ou rejeitada.

## Modelos

`writers.json` é a lista de escritores. Um uso é permitido, a menos que uma cláusula na licença ou nos termos do criador o restrinja; o silêncio não é uma restrição. Cada entrada registra a pesquisa de termos: a URL da página, a data em que foi lida, quem a leu, seu `registry_id` no registro de licenças da rnd (`licences/registry.json`) e seu veredicto de treinamento (`allowed`, `conditions` ou `barred`). Uma entrada sem esse registro, ou cujo veredicto seja `barred`, é rejeitada quando a lista é carregada. Um nome que não está na lista é rejeitado, assim como um nome que contém `cloud`.

Os escritores são Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B (estes três em um pod) e os oito braços do piloto: Nemotron 3.5 Bolt 74B (a construção própria da NVIDIA de 24 GB com reconhecimento de quantização), Muse-Glimmer-30B, Nemotron 3.5 Lightning 30B, Qwen3.8-27B, Qwen3.6-35B-A3B, Granite-4.2-30B, Gemma-4-31B-it e DeepSeek-R1-Distill-Qwen-32B. Os críticos são Muse-Glimmer-30B e Nemotron 3.5 Lightning. Um crítico da mesma família do escritor é rejeitado. Cada entrada fixada contém um `pin_note`: como seu GGUF foi criado e se ele se encaixa em um 40 GB A100 no contexto de 32.768 tokens do piloto.

Os modelos são servidos pelo `llama-server` do llama.cpp, construído com CUDA 13.4, em Q8_0. Um modelo listado é rejeitado até que seja fixado: sua revisão do repositório, arquivo GGUF e o sha256 do arquivo. O script wave calcula o hash do GGUF e do binário `llama-server` antes do lançamento, e a execução rejeita qualquer desvio da lista. Cada recibo registra a partição (`production` ou `trial:<name>`), a família, a revisão e o hash GGUF de ambos os modelos, o commit e o hash do servidor e a linha da máquina CUDA.

**Valores dos cartões.** O escritor e o crítico são executados com as configurações documentadas do seu criador, lidas a partir de `experiments/local-builders/card-values.json` com `--cards` (o arquivo J2 é lido, portanto, um modelo é executado da mesma forma em ambos). Cada chave de amostragem e as configurações do modelo de chat são aplicadas em cada chamada, incluindo os valores desativados, para que os valores padrão de `llama-server` nunca cheguem a um modelo. O servidor é executado com um contexto de 32.768 tokens e uma resposta pode usar 16.384, já que cada modelo com um modo de raciocínio pensa. O texto do sistema R1-Distill é incorporado à sua vez de usuário, conforme solicitado pelo seu cartão. Um escritor é associado ao seu cartão pelo seu ID `card` em `writers.json` ou pelo seu nome de serviço. Os registros armazenam o sha256 do arquivo, o candidato e exatamente o que foi enviado, para o escritor e o crítico. Um `write` ou `critic-control` real sem `--cards` é rejeitado.

Uma etapa é executada apenas dentro de uma concessão, a menos que sua linha de máquina nomeie uma VM do Colab ou um pod: `--grant` e um arquivo de sinalização modificado nos últimos 60 segundos. Uma linha que nomeia o equipamento precisa deles, independentemente do que seu cartão diga. O arquivo de sinalização não é aberto. `--mock` não carrega nada.

## Testes piloto

Três testes piloto pagos são registrados em [docs/runs/](docs/runs/), todos em 11 de outubro de 2026: US$ 3,52, US$ 3,61 e US$ 2,80. Estes são os únicos valores de custo que este README declara. Nenhuma linha deles está em um conjunto de treinamento. No teste piloto 2, a verificação de 49 candidatos da equipe de P&D aprovou 15 conforme escrito, mais 15 após uma correção mecânica e reprovou 19. O teste piloto 3 não escreveu nada.

## Modelo de ameaças

role-set é um gerador local. Não é um servidor. Não escuta e não coleta dados de telemetria.

Uma execução afeta o plano, o pacote de funções e o diretório de saída que você nomeia. `ingest` e `store` também afetam o arquivo de armazenamento que você nomeia. `write`, `pilot-auto` e `critic-control` afetam um endpoint de loopback quando não são `--mock`. `--mock` não envia nada. O gerador e seus scripts leem e gravam arquivos dentro do repositório e não chamam nenhum modelo.

Uma execução não afeta um serviço de modelo em nuvem, um pod, uma conta de aluguel ou uma chave de API de aluguel. Não lê linhas seladas, conjuntos retidos ou sessões de entrevista. `ingest` lê as doze perguntas públicas da entrevista do clone aspire-si que você nomeia, no commit que você nomeia, apenas para rejeitar uma linha que as repita. Elas não são armazenadas. `store status` retorna contagens, não linhas. Relate uma vulnerabilidade por meio do contato em [SECURITY.md](SECURITY.md).

## Testes

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

A cobertura de linhas tem um limite de 90, em `pyproject.toml` e `codecov.yml`, e a execução falha abaixo desse limite. Ela mede cerca de 97. O selo de cobertura acima mostra o valor atual. O CI também executa a verificação do atlas, `role-set plan` e `puzzle-check` em um fixture, sem modelo. `scripts/verify` executa as verificações locais em um único comando.

## Colaboradores

Partes deste repositório foram escritas com Grok (xAI). Grok, Argon e Claude são creditados em [CONTRIBUTORS.md](CONTRIBUTORS.md).

O manual é publicado com a [página de destino](https://mcp-tool-shop-org.github.io/role-set/). Este README é independente.

Criado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
