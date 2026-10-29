<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototipo, desarrollado de forma abierta.** Consultivo y experimental: las interfaces y las estructuras de datos cambian sin previo aviso. Nada de lo que se define en el conjunto de roles se considera datos de entrenamiento hasta que una persona lo revise. Cada elemento pasa por la verificación clave del departamento de I+D y por una segunda revisión antes de acercarse a un conjunto de entrenamiento.

El plan del conjunto de roles define los elementos que conforman el rol, escribe las partes que debe escribir el autor y registra las etapas por las que pasó cada elemento. También crea rompecabezas de tipo B en código y los prueba antes de que un modelo los vea. Es un generador, no un juez. Lo que define a un buen responsable de rol se encuentra en un paquete de roles, que crean las personas.

El autor y el crítico solo se pueden contactar a través de `http://127.0.0.1` con un puerto explícito. Cualquier otra cosa se rechaza antes de que se envíe un byte. El conjunto de roles nunca alquila una máquina ni manipula un pod. Cuando una ejecución utiliza un pod, Offrig lo alquila y el conjunto de roles solo ve el túnel de bucle.

Los diseños se encuentran en [docs/role-set-design.md](docs/role-set-design.md) y [docs/generator-design.md](docs/generator-design.md), y se mantienen en línea con el programa. Las notas de la ejecución se encuentran en [docs/runs/](docs/runs/).

## Instalación

Python 3.13 o posterior (3.15 es el objetivo principal; las pruebas de CI prueban las versiones 3.13, 3.14 y 3.15). Una dependencia de tiempo de ejecución: [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 1.7.6, fijada. Las etapas se importan, no se copian.

```text
pip install role-set
```

Para trabajar en el propio conjunto de roles, clónelo y deje que uv instale el entorno bloqueado:

```text
uv sync --frozen
```

## Los paquetes de roles

Un paquete de roles es el texto que las personas escribieron para un rol. El conjunto de roles lo sirve en fragmentos y no añade nada a él. Su hash se encuentra en cada recibo aceptado.

- **El verificador** (`roles/verifier`). `pack sync` lo copia de un clon local de aspire-si en un commit: las lecciones, los 18 rasgos, la tabla de tipos, las enmiendas y las reglas. `pack verify` comprueba la copia con su manifiesto. Un clon nuevo solo contiene la lista de fuentes, la lista de exclusiones y `defects.jsonl`, el catálogo de clases de defectos que encontraron las revisiones. La sincronización completa el resto.
- **El defensor** (`roles/advocate`). Un paquete que una persona escribió en este repositorio, sin ningún texto de modelo en él. El director aprobó el paquete el 14 de octubre de 2026 (commit `ffb9546` en la solicitud de extracción n.º 56) y el lote 1 se está creando allí; no estará en `main` hasta que se fusionen los cambios de la solicitud n.º 56. El lote 1 tiene tres rasgos: `failed-gate-first`, `exact-figures` y `faithful-relay`. Ningún elemento del defensor se considera datos de entrenamiento hasta que pasa la lectura de prueba.

## El generador T10

Los rompecabezas de tipo B solían escribirse a mano alzada y luego se verificaban. Ahora, el código crea el rompecabezas y lo prueba, y un modelo solo lo formula. El generador es el paquete `role_set.generate`. T10, la lección sobre palabras de tamaño estricto ("exactamente", "al menos", "como máximo"), es la parte que cambió en la versión 0.2.0. También se crea una receta T12 (qué edición estaba en vigor en una fecha). T11 no se crea.

**Comprobado mediante la construcción.** El creador prueba una solución oculta, añade pistas que son verdaderas en ella hasta que se encuentra la respuesta, y elimina todas las pistas que la respuesta no necesita. El solucionador comprueba el modelo. A continuación, el creador ejecuta las cuatro pruebas T10 de I+D: «exactamente» se interpreta como «al menos», «exactamente» se interpreta como «como máximo», «al menos» se interpreta como «exactamente» y «como máximo» se interpreta como «exactamente». Un acertijo se conserva solo cuando una de estas interpretaciones, aplicada únicamente a la pista del rasgo, cambia la respuesta. Un acertijo que se resuelve basándose en alguna otra pista numérica pertenece a la clase de defectos `trait_not_tested` y se descarta.

**El mapa de celdas.** Cada paquete, pregunta y semilla se asigna a una celda estructural: el recuento de entidades `n` (de 3 a 5), la palabra de recuento `op`, `k` (de 1 a n−1) y `ifs`, el número de pistas `if` (de 0 a 2). Las preguntas de conjunto utilizan las tres palabras de recuento. Las preguntas de afirmación solo utilizan "exactamente", porque "al menos" y "como máximo" no pueden cambiar el veredicto de una afirmación. Las celdas utilizables se exploran una vez y se confirman en `t10_cells.json`.

**El eje de afirmación `if_shape`.** Una celda de afirmación también nombra la forma de sus `if` pistas. Hay siete formas: `none` para ninguna `if`; `pos` o `neg` para una `if`, según la polaridad de su consecuente; y `chain`, `shared_consequent`, `shared_antecedent` o `disjoint` para dos. Una construcción cuyo rompecabezas reducido pierde su forma vuelve a intentarlo dentro de la misma semilla.

**El asignador.** Para una pregunta y un recuento de valores, se asigna un archivo de 20 semillas una vez, antes de que se cree algo, mediante un flujo de costo mínimo. Las 20 semillas de cada paquete ocupan 20 celdas diferentes. Una celda que no tiene ninguna pista `if` se utiliza como máximo dos veces en un archivo (la restricción de cero si es necesario). Las visitas posteriores favorecen las celdas que dieron más esqueletos distintos en las semillas de ajuste. Esos pesos se mantienen separados para los paquetes de dos y tres valores.

**Los límites de variedad.** En cada estrato (conjunto y afirmación, dos y tres valores): al menos el 75% de esqueletos canónicos distintos, ningún esqueleto más de 3 veces y ningún esqueleto dos veces en un archivo de un paquete. Los límites se juzgan en semillas bloqueadas. En las semillas bloqueadas 101 a 120, las afirmaciones de tres valores alcanzan 118/120 distintos, con una repetición máxima de 2.

**El registro confirmado.** `t10_record.json` contiene 1280 construcciones (semillas 1–20, 61–80, 81–100 y 101–120, cada paquete, ambas preguntas), cada una con su celda, `if_shape`, intentos y esqueleto canónico. Contiene un sha256 sobre la fuente del generador, la tabla de celdas, la tabla de capacidad y los datos del paquete. Una prueba mantiene esa lista de archivos cerrada bajo las propias importaciones del conjunto de roles. El trabajo normal de prueba de CI comprueba el hash y los límites del registro en minutos. La reconstrucción exhaustiva de cada rompecabezas se ejecuta como el trabajo `sweep`, solo en `workflow_dispatch`, y falla si hay alguna diferencia con el registro.

**El límite de capacidad y el espacio disponible.** `scripts/t10_usable_cells.py --check` muestra la oferta, la demanda y el espacio disponible para cada estrato, y falla si no se puede colocar un archivo. Con `if_shape`, los paquetes de tres valores tienen 61 celdas utilizables entre ellos, en comparación con las 27 anteriores. Ningún paquete necesita una visita con valor cero-`if`, y una prueba coloca doce paquetes de tres valores (los seis reales y seis copias) sin repetición dentro de un archivo.

## Comandos

`role-set --help` enumera los comandos. Cada línea que se muestra a continuación es un comando que existe en la actualidad. Los indicadores se encuentran en el `--help` de cada comando.

| Comando | Qué hace |
|---|---|
| `plan` | Corrige cada ranura para un lote a partir de una semilla. No se realiza ninguna llamada al modelo. La misma semilla genera el mismo plan. |
| `cost` | Estima las horas de funcionamiento y el coste de los pods para un plan antes de que se inicie cualquier cosa (`two-card` o `frontier`). Es una estimación. |
| `pack sync` / `pack verify` | Copia el paquete de verificación de un clon local de aspire-si en un commit específico y verifica la copia con su manifiesto. |
| `harvest` | Copia fragmentos con licencia verificada de repositorios fijados a un grupo local. No se registra el directorio del clon. |
| `write` | El ejecutor. `--mock` lo ejecuta sin modelo y sin coste. Una ejecución real necesita un bucle `--endpoint` y un `--model` de `writers.json`. `--wave write`, `critique` y `rewrite` dividen un lote para que el escritor y el crítico se ejecuten uno a la vez, en diferentes hosts si es necesario, con `wave-drafts.jsonl` entre ellos. |
| `readout` | Informa sobre una ejecución por lección y tipo, indicando si tuvo éxito o no, a partir del directorio que contiene `batch-report.json`. |
| `critic-control` | Mide el rendimiento de un crítico en los elementos revisados, en su servidor llama. Necesita los hashes GGUF y del servidor del crítico, así como la línea de la máquina, además de una autorización y un archivo de señal de actividad de un host alquilado. `--trait-leg` y `--answer-leg` ejecutan esas etapas. |
| `ingest` / `store` | Registra los recibos en el almacén, con el firewall en la puerta. `store status` informa sobre los recuentos. `store void` añade una entrada nula. |
| `pilot-auto` | El lanzador pagado de los pilotos de 2026-10-11. Su script nombra los repositorios Qwen y Gemma de los pilotos, que no están en `writers.json`, por lo que `write` lo rechaza. |
| `puzzle-check` | Ejecuta el solucionador de tipo B sobre archivos de datos y sus `.models.jsonl` archivos adjuntos, y rechaza un esquema de hechos repetido. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

El generador aún no tiene el comando `role-set`. Sus tres scripts crean las tablas confirmadas y el registro:

| Script | Qué hace |
|---|---|
| `scripts/t10_usable_cells.py` | Examina cada celda y escribe `t10_cells.json`. `--check` ejecuta la puerta de capacidad en la tabla confirmada y no examina. |
| `scripts/t10_cell_capacity.py` | Cuenta los esquemas distintos por celda en las semillas de ajuste 1, 21 y 41, por estrato, en `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Reconstruye cada rompecabezas registrado en `t10_record.json`. `--check` verifica el registro confirmado e imprime las cifras de la semilla bloqueada. |

## Puertas

Cinco tipos:

- **A.** Un pasaje de enseñanza. Sin veredicto. Se ejecuta un validador y la comprobación de características es meramente informativa.
- **B.** Un rompecabezas de asignación finita. El solucionador verifica el modelo. La respuesta del crítico es meramente informativa.
- **C.** Un verificador con un fallo intencionado o un verificador que acertó. Las comprobaciones de veredicto y evidencia se rechazan. La comprobación de fallos se registra y no se rechaza.
- **D.** Dos turnos. Las comprobaciones de veredicto y evidencia se rechazan. La comprobación de características es meramente informativa.
- **E.** Un cuestionario escrito o un cuestionario de veredicto. Un E escrito no tiene comprobación de veredicto. Un E de tipo veredicto verifica el veredicto y la evidencia.

Las comprobaciones de veredicto y evidencia se rechazan. Las comprobaciones de características, respuestas y fallos se registran y no se rechazan, hasta que sus controles se superen. La etapa de características midió un 40,3% en general, frente a una barra del 85%, y la peor característica, un 12,5% frente al 70%. La etapa de respuesta midió un 62,7% frente a una barra del 90%. El control de fallos del 2026-10-11 tenía 9/24.

El solucionador B cubre rompecabezas de asignación finita en su conjunto cerrado de formas de reglas: como máximo 6 entidades, 3 atributos y 200.000 asignaciones. No verifica el razonamiento en relación con el modelo. Un elemento B con clave de veredicto sobre material de texto libre está fuera del alcance del solucionador, y `puzzle-check` lo cuenta como tal, nunca como superado.

El firewall del almacén rechaza el material retenido, sellado y de entrevistas durante la ingestión, y el material DEV durante una etapa. No hay ninguna ruta de lectura para ese material. La dirección se ha diseñado y nunca se ha ejecutado.

## Qué guarda una ejecución

Cada recibo aceptado contiene el hash del plan y el hash del paquete. Cuando la ejecución utilizó ejemplos, también contiene el commit del ejemplo y el hash del archivo de exclusión. Cuando utilizó la regla bloqueada, contiene el hash de esa regla. El recibo de un rompecabezas generado contiene su semilla, intentos, lecturas, métricas de dificultad y los hashes del esquema, el paquete y el solucionador. Cada llamada al escritor se guarda en `drafts.jsonl`, ya sea aceptada o rechazada.

## Modelos

`writers.json` es la lista de escritores. Se permite un uso a menos que una cláusula en la licencia o los términos del creador lo restrinja; el silencio no es una restricción. Cada entrada registra la búsqueda de términos: la URL de la página, la fecha en que se leyó, quién la leyó, su `registry_id` en el registro de licencias de rnd (`licences/registry.json`) y su veredicto de entrenamiento (`allowed`, `conditions` o `barred`). Se rechaza una entrada sin ese registro, o cuyo veredicto sea `barred`, cuando se carga la lista. Se rechaza un nombre que no esté en la lista, así como un nombre que contenga `cloud`.

Los escritores son Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B (estos tres en un pod), y los ocho brazos del piloto: Nemotron 3.5 Bolt 74B (la propia versión de NVIDIA de 24 GB con cuantificación consciente), Muse-Glimmer-30B, Nemotron 3.5 Lightning 30B, Qwen3.8-27B, Qwen3.6-35B-A3B, Granite-4.2-30B, Gemma-4-31B-it y DeepSeek-R1-Distill-Qwen-32B. Los críticos son Muse-Glimmer-30B y Nemotron 3.5 Lightning. Se rechaza un crítico de la misma familia que el escritor. Cada entrada fijada contiene un `pin_note`: cómo se creó su GGUF y si se ajusta a un A100 de 40 GB en el contexto de 32.768 tokens del piloto.

Los modelos son servidos por `llama-server` de llama.cpp, compilado con CUDA 13.4, en Q8_0. Se rechaza un modelo listado hasta que se fija: su revisión del repositorio, el archivo GGUF y el sha256 del archivo. El script wave calcula el hash del GGUF y del binario `llama-server` antes del lanzamiento, y la ejecución rechaza cualquier desviación de la lista. Cada recibo registra la partición (`production` o `trial:<name>`), la familia, la revisión y el hash GGUF de ambos modelos, el commit y el hash del servidor, y la línea de la máquina CUDA.

**Valores de las tarjetas.** El escritor y el crítico se ejecutan con la configuración documentada de su creador, que se lee desde `experiments/local-builders/card-values.json` con `--cards` (el archivo J2 se lee, por lo que un modelo se ejecuta de la misma manera en ambos). Cada clave de muestreo y la configuración de la plantilla de chat se aplican en cada llamada, incluidos los valores desactivados, por lo que los valores predeterminados de `llama-server` nunca llegan a un modelo. El servidor se ejecuta con un contexto de 32.768 tokens y una respuesta puede usar 16.384, ya que cada modelo con un modo de razonamiento piensa. El texto del sistema de R1-Distill se incluye en su turno de usuario, como lo indica su tarjeta. Un escritor se asigna a su tarjeta mediante su ID `card` en `writers.json`, o mediante su nombre de servicio. Los registros registran el sha256 del archivo, el candidato y exactamente lo que se envió, para el escritor y el crítico. Se rechaza un `write` o `critic-control` real sin `--cards`.

Una etapa se ejecuta solo dentro de una asignación, a menos que su línea de máquina especifique una máquina virtual de Colab o un pod: `--grant` y un archivo de latido que se modificó hace menos de 60 segundos. Una línea que especifica el equipo necesita estos archivos, independientemente de lo que diga su tarjeta. El archivo de latido no se abre. `--mock` no carga nada.

## Pruebas piloto

Se registran tres pruebas piloto pagadas en [docs/runs/](docs/runs/), todas del 11 de octubre de 2026: 3,52 $, 3,61 $ y 2,80 $. Estos son los únicos datos de costos que indica este archivo README. Ninguna fila de ellos está en un conjunto de entrenamiento. En la prueba piloto 2, la verificación de claves de I+D de 49 candidatos aprobó 15 según lo escrito, 15 más después de una corrección mecánica y falló 19. La prueba piloto 3 no escribió nada.

## Modelo de amenazas

role-set es un generador local. No es un servidor. No escucha y no recopila datos de telemetría.

Una ejecución afecta al plan, al paquete de roles y al directorio de salida que especifique. `ingest` y `store` también afectan al archivo de almacenamiento que especifique. `write`, `pilot-auto` y `critic-control` afectan a un punto final de bucle cuando no son `--mock`. `--mock` no envía nada. El generador y sus scripts leen y escriben archivos dentro del repositorio y no llaman a ningún modelo.

Una ejecución no afecta a un servicio de modelo en la nube, a un pod, a una cuenta de alquiler o a una clave de API de alquiler. No lee filas selladas, conjuntos excluidos o sesiones de entrevistas. `ingest` lee las doce preguntas públicas de la entrevista del clon aspire-si que especifique, en el commit que especifique, solo para rechazar una fila que las repita. No se almacenan. `store status` devuelve recuentos, no filas. Informe de una vulnerabilidad a través del contacto en [SECURITY.md](SECURITY.md).

## Pruebas

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La cobertura de líneas tiene un mínimo del 90, en `pyproject.toml` y `codecov.yml`, y la ejecución falla si está por debajo de este valor. Mide aproximadamente el 97%. La insignia de cobertura que se muestra arriba muestra la cifra actual. CI también ejecuta atlas check, `role-set plan` y `puzzle-check` en un elemento de prueba, sin ningún modelo. `scripts/verify` ejecuta las comprobaciones locales en un solo comando.

## Colaboradores

Partes de este repositorio se escribieron con Grok (xAI). Se reconoce a Grok, Argon y Claude en [CONTRIBUTORS.md](CONTRIBUTORS.md).

El manual se publica con la [página de inicio](https://mcp-tool-shop-org.github.io/role-set/). Este archivo README se mantiene por sí solo.

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
