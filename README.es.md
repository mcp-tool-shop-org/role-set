<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototipo, desarrollado de forma abierta.** Consultivo y experimental: las interfaces y las estructuras de datos cambian sin previo aviso. Ninguna de las indicaciones establecidas se considera datos de entrenamiento hasta que una persona las revise. Cada elemento pasa por la verificación clave del departamento de I+D y una segunda revisión antes de acercarse a un conjunto de entrenamiento.

`role-set` planifica los elementos de definición de roles, escribe las partes que debe escribir un redactor y registra las etapas por las que pasó cada elemento. También crea rompecabezas de tipo B en código y los verifica antes de que un modelo los vea. Es un generador, no un juez. Lo que define a un buen responsable de roles reside en un paquete de roles, que crean las personas.

El redactor y el crítico solo se pueden contactar en `http://127.0.0.1` con un puerto explícito. Cualquier otra cosa se rechaza antes de que se envíe un solo byte. `role-set` nunca alquila una máquina ni manipula un pod. Cuando una ejecución utiliza un pod, Offrig lo alquila y `role-set` solo ve el túnel de bucle.

Los diseños son [docs/role-set-design.md](docs/role-set-design.md) y [docs/generator-design.md](docs/generator-design.md), y se mantienen en consonancia con el programa. Las notas de la ejecución se encuentran en [docs/runs/](docs/runs/).

## Instalación

Python 3.12, gestionado por uv. Una dependencia de tiempo de ejecución: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si) 1.4.0, fijada a la confirmación de aspire-si `388a6bc`. Las etapas se importan, no se copian.

```text
uv sync --frozen
```

## Los paquetes de roles

Un paquete de roles es el texto que las personas escribieron para un rol. `role-set` lo sirve en fragmentos y no añade nada a él. Su hash está en cada recibo aceptado.

- **El verificador** (`roles/verifier`). `pack sync` lo copia de un clon local de aspire-si en una confirmación: las lecciones, los 18 rasgos, la tabla de tipos, las enmiendas y las reglas. `pack verify` verifica la copia con su manifiesto. Un clon nuevo solo contiene la lista de origen, la lista de exclusión y `defects.jsonl`, el catálogo de clases de defectos que encontraron las revisiones. La sincronización completa el resto.
- **El defensor** (`roles/advocate`). Un paquete que una persona escribió en este repositorio, sin ningún texto de modelo en él. El director aprobó el paquete el 14 de octubre de 2026 (confirmación `ffb9546` en la solicitud de extracción n.º 56), y el lote 1 se está creando allí; no estará en `main` hasta que se fusionen los cambios de la solicitud n.º 56. El lote 1 tiene tres rasgos: `failed-gate-first`, `exact-figures` y `faithful-relay`. Ningún elemento del defensor es un dato de entrenamiento hasta que pasa la revisión piloto.

## El generador T10

Los rompecabezas de tipo B solían escribirse a mano alzada y luego se verificaban. Ahora, el código crea el rompecabezas y lo verifica, y un modelo solo lo formula. El generador es el paquete `role_set.generate`. T10, la lección sobre palabras de tamaño estricto ("exactamente", "al menos", "como máximo"), es la parte que cambió en la versión 0.2.0. También se crea una receta T12 (qué edición estaba en vigor en una fecha). T11 no.

**Verificado por construcción.** El creador muestrea una solución oculta, añade pistas que son verdaderas en ella hasta que se fija la respuesta y elimina todas las pistas que la respuesta no necesita. El solucionador verifica el modelo. Luego, el creador ejecuta las cuatro lecturas T10 del departamento de I+D: "exactamente" se lee como "al menos", "exactamente" se lee como "como máximo", "al menos" se lee como "exactamente" y "como máximo" se lee como "exactamente". Un rompecabezas se conserva solo cuando una de esas lecturas, aplicada a la pista del rasgo por sí sola, cambia la respuesta. Un rompecabezas que pasa en alguna otra pista de recuento es la clase de defecto `trait_not_tested`, y se rechaza.

**El mapa de celdas.** Cada paquete, pregunta y semilla se asigna a una celda estructural: el recuento de entidades `n` (de 3 a 5), la palabra de recuento `op`, `k` (de 1 a n-1) y `ifs`, el número de pistas `if` (de 0 a 2). Las preguntas de conjunto utilizan las tres palabras de recuento. Las preguntas de afirmación solo utilizan "exactamente", porque "al menos" y "como máximo" no pueden cambiar el veredicto de una afirmación. Las celdas utilizables se exploran una vez y se confirman en `t10_cells.json`.

**El eje de afirmación `if_shape`.** Una celda de afirmación también nombra la forma de sus pistas `if`. Hay siete formas: `none` para ninguna pista `if`; `pos` o `neg` para una pista `if`, según la polaridad de su consecuente; y `chain`, `shared_consequent`, `shared_antecedent` o `disjoint` para dos pistas. Una construcción cuyo rompecabezas reducido pierde su forma vuelve a intentarlo dentro de la misma semilla.

**El asignador.** Para una pregunta y un recuento de valores, se asigna un archivo de 20 semillas una vez, antes de que se construya algo, mediante un flujo de costo mínimo. Las 20 semillas de cada paquete ocupan 20 celdas diferentes. Una celda sin ninguna pista `if` se utiliza como máximo dos veces en un archivo (el límite de cero si). Las visitas posteriores favorecen las celdas que dieron más esqueletos distintos en las semillas de ajuste. Esos pesos se mantienen separados para los paquetes de dos y tres valores.

**Los límites de variedad.** En cada estrato (conjunto y afirmación, dos y tres valores): al menos el 75% de esqueletos canónicos distintos, ningún esqueleto más de 3 veces y ningún esqueleto dos veces en un archivo del paquete. Los límites se juzgan en semillas bloqueadas. En las semillas bloqueadas 101 a 120, las afirmaciones de tres valores alcanzan 118 de 120 distintos, con una repetición máxima de 2.

**El registro confirmado.** `t10_record.json` contiene 1280 construcciones (semillas 1–20, 61–80, 81–100 y 101–120, cada paquete, ambas preguntas), cada una con su celda, `if_shape`, intentos y esqueleto canónico. Contiene un sha256 sobre el código fuente del generador, la tabla de celdas, la tabla de capacidad y los datos del paquete. Una prueba mantiene esa lista de archivos cerrada bajo las propias importaciones de `role-set`. El trabajo normal de prueba de CI verifica el hash y los límites del registro en minutos. La reconstrucción exhaustiva de cada rompecabezas se ejecuta como el trabajo `sweep`, solo en `workflow_dispatch`, y falla si hay alguna diferencia con el registro.

**La puerta de capacidad y el margen.** `scripts/t10_usable_cells.py --check` informa sobre la oferta, la demanda y el margen para cada estrato, y falla con los números cuando un archivo no se puede colocar. Con `if_shape`, los paquetes de tres valores tienen 61 celdas de afirmación utilizables entre ellos, en comparación con las 27 anteriores. Ningún paquete necesita una visita de cero si `if`, y una prueba coloca doce paquetes de tres valores (los seis reales y seis clones) sin ninguna repetición dentro de un archivo.

## Comandos

`role-set --help` enumera los comandos. Cada línea a continuación es un comando que existe en la actualidad. Las opciones están en el `--help` de cada comando.

| Comando | Qué hace |
|---|---|
| `plan` | Corrige cada ranura para un lote a partir de una semilla. No se realiza ninguna llamada al modelo. La misma semilla escribe el mismo plan. |
| `cost` | Se estiman las horas de trabajo y el coste en dólares para un plan antes de que se inicie cualquier cosa (`two-card` o `frontier`). Es una estimación. |
| `pack sync` / `pack verify` | Copia el paquete de verificación de un clon local de aspire-si en un commit determinado y verifica la copia con su manifiesto. |
| `harvest` | Copia fragmentos con licencia verificada de repositorios fijados a un grupo local. No se registra el directorio del clon. |
| `write` | El ejecutor. `--mock` lo ejecuta sin modelo ni coste. Una ejecución real necesita un bucle `--endpoint` y un `--model` de `writers.json`. |
| `readout` | Informa de una ejecución por lección y tipo, indicando si tuvo éxito o no, a partir del directorio que contiene `batch-report.json`. |
| `critic-control` | Mide el crítico en los elementos revisados. Necesita una subvención y un archivo de señal de vida. `--trait-leg` y `--answer-leg` ejecutan esas etapas. |
| `ingest` / `store` | Registra los recibos en el almacén, con el firewall en la puerta. `store status` informa de los recuentos. `store void` añade una entrada nula. |
| `pilot-auto` | El lanzador de pago: espera la etapa de bucle, escribe y puede volver a iniciarse después de una interrupción por llamada perdida. |
| `puzzle-check` | Ejecuta el solucionador de tipo B sobre archivos de datos y sus archivos auxiliares `.models.jsonl`, y rechaza un esquema de hechos repetido. |

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

- **A.** Un pasaje de enseñanza. Sin veredicto. Se ejecuta un validador y la verificación de características es meramente informativa.
- **B.** Un rompecabezas de asignación finita. El solucionador verifica el modelo. La respuesta del crítico es meramente informativa.
- **C.** Un verificador con un fallo intencionado o un verificador que acertó. Las verificaciones de veredicto y evidencia rechazan. La verificación de fallos se registra y no rechaza.
- **D.** Dos turnos. Las verificaciones de veredicto y evidencia rechazan. La verificación de características es meramente informativa.
- **E.** Un cuestionario escrito o un cuestionario de veredicto. Un E escrito no tiene verificación de veredicto. Un E de tipo veredicto verifica el veredicto y la evidencia.

Las verificaciones de veredicto y evidencia rechazan. Las verificaciones de características, respuestas y fallos se registran y no rechazan, hasta que sus controles tengan éxito. La etapa de características midió un 40,3% en general, frente a una barra del 85%, y la peor característica, un 12,5% frente al 70%. La etapa de respuesta midió un 62,7% frente a una barra del 90%. El control de fallos del 2026-10-11 tuvo 9 de 24.

El solucionador B cubre rompecabezas de asignación finita en su conjunto cerrado de formas de reglas: como máximo 6 entidades, 3 atributos y 200.000 asignaciones. No verifica el razonamiento con el modelo. Un elemento B con clave de veredicto sobre material de texto libre está fuera del solucionador, y `puzzle-check` lo cuenta como tal, nunca como superado.

El firewall del almacén rechaza el material retenido, sellado y de entrevistas al ingerirse, y el material de DEV durante una etapa. No hay ninguna ruta de lectura para ese material. La dirección se ha diseñado y nunca se ha ejecutado.

## Lo que guarda una ejecución

Cada recibo aceptado contiene el hash del plan y el hash del paquete. Cuando la ejecución utilizó ejemplos, también contiene el commit del ejemplo y el hash del archivo de exclusión. Cuando utilizó la regla bloqueada, contiene el hash de esa regla. El recibo de un rompecabezas generado contiene su semilla, intentos, lecturas, métricas de dificultad y los hashes del esquema, paquete y solucionador. Cada llamada de escritura se guarda en `drafts.jsonl`, ya sea aceptada o rechazada.

## Modelos

El pin del escritor es Qwen3-235B-A22B-Instruct-2507, la versión AWQ, Apache-2.0, que se utiliza como `qwen3-235b-instruct-2507` en la revisión `96548f657d772187f16e9f04bc50aec78a9f9cdd`. El pin del crítico es `gemma4:31b`, resumen `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, Q4_K_M, Apache-2.0, en `http://127.0.0.1:11434`. `writers.json` también enumera un escritor solo para ensayos y un crítico de comparación. Se rechaza un nombre que no esté en el archivo, así como un nombre que contenga `cloud`.

La GPU local se utiliza para el crítico y un escritor de ensayo, solo dentro de una subvención: `--grant` y un archivo de señal de vida modificado hace menos de 60 segundos. El archivo de señal de vida no se abre. Un crítico que la ejecución cargó se descarga al salir, incluso en caso de error. `--mock` no carga nada.

## Pilotos

Se registran tres pilotos de pago en [docs/runs/](docs/runs/), todos con fecha del 2026-10-11: 3,52 $, 3,61 $ y 2,80 $. Estas son las únicas cifras de coste que indica este archivo LÉAME. Ninguna fila de ellos está en un conjunto de entrenamiento. En el piloto 2, la verificación clave de 49 candidatos de I+D aprobó 15 tal cual, 15 más después de una corrección mecánica y rechazó 19. El piloto 3 no escribió nada.

## Modelo de amenazas

role-set es un generador local. No es un servidor. No escucha y no recopila datos de telemetría.

Una ejecución afecta al plan, al paquete de roles y al directorio de salida que especifique. `ingest` y `store` también afectan al archivo del almacén que especifique. `write`, `pilot-auto` y `critic-control` afectan a un punto final de bucle cuando no son `--mock`. `--mock` no envía nada. El generador y sus scripts leen y escriben archivos dentro del repositorio y no llaman a ningún modelo.

Una ejecución no afecta a un servicio de modelo en la nube, a un pod, a una cuenta de alquiler ni a una clave de API de alquiler. No lee filas selladas, conjuntos retenidos ni sesiones de entrevistas. `ingest` lee las doce preguntas públicas de la entrevista del clon de aspire-si que especifique, en el commit que especifique, solo para rechazar una fila que las repita. No se almacenan. `store status` devuelve recuentos, no filas. Informe de una vulnerabilidad a través del contacto en [SECURITY.md](SECURITY.md).

## Pruebas

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La cobertura de líneas tiene un mínimo del 90, en `pyproject.toml` y `codecov.yml`, y la ejecución falla si está por debajo. Mide aproximadamente el 97%. La insignia de cobertura que se muestra arriba indica la cifra actual. CI también ejecuta la comprobación de atlas, `role-set plan` y `puzzle-check` en un elemento de prueba, sin modelo. `scripts/verify` ejecuta las comprobaciones locales en un solo comando.

El manual se publica con la [página de destino](https://mcp-tool-shop-org.github.io/role-set/). Este archivo LÉAME es independiente.

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
