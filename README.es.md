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

**Prototipo, desarrollado de forma abierta.** Consultivo y experimental: las interfaces y las estructuras de datos cambian sin previo aviso.

El conjunto de roles planifica los elementos de configuración de roles, escribe las partes que un escritor debe escribir y registra qué puertas atravesó cada elemento. Es un generador, no un juez. Lo que define a un buen verificador reside en el paquete de roles, que los humanos escriben. Nada de lo que escribe este programa está en un conjunto de entrenamiento. Cada elemento pasa por la verificación clave de I+D y una segunda revisión antes de acercarse a una de ellas.

El escritor y el crítico solo se pueden contactar en `http://127.0.0.1` con un puerto explícito. Cualquier otra cosa se rechaza antes de que se envíe un byte. El conjunto de roles nunca alquila una máquina ni manipula un pod. Cuando una ejecución utiliza un pod, offrig es quien lo alquila, y el conjunto de roles solo ve el túnel de bucle.

El diseño se encuentra en [docs/role-set-design.md](docs/role-set-design.md) y se ha mantenido coherente con el programa. Las notas sobre la ejecución se encuentran en [docs/runs/](docs/runs/).

## Instalación

Python 3.12, gestionado por uv. Una dependencia de tiempo de ejecución: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si), fijada a un commit de aspire-si. Las puertas se importan, no se copian.

```text
uv sync --frozen
```

## Comandos

`role-set --help` enumera los comandos. Cada línea a continuación es un comando que existe en la actualidad.

`plan` corrige cada ranura para un lote a partir de una semilla. No llama a un modelo. La misma semilla reproduce el plan.

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` estima las horas de pod y los dólares para ese plan antes de que se inicie algo. La cifra es una estimación. Los perfiles son `two-card` y `frontier`.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` copia el paquete de roles desde un clon local de aspire-si en un commit. `pack verify` verifica esa copia con su manifiesto. La ruta del clon no se registra. Un clon nuevo de este repositorio no es un paquete sincronizado: `roles/verifier` contiene la lista de exclusión y la lista de origen, y la sincronización completa el resto.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` copia extractos con licencia verificada en un grupo local. El directorio del clon no se registra.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` es el ejecutor. `--mock` ejecuta toda la canalización sin modelo y sin gasto, y no solicita una subvención. La salida simulada está marcada, no es un dato y se escribe en `<out>/mock/`. Una ejecución que llama a un modelo necesita `--endpoint` y `--model`, y el punto final debe ser de bucle. Cargar el crítico, o un escritor de ensayo local, también necesita `--grant` y `--heartbeat`. `--no-critic` es una ejecución de prueba: los elementos pasan sin el segundo lector, y el escritor del pod no utiliza la GPU local.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` informa de una ejecución por lección y tipo, incluida una ejecución que no aceptó nada. Lee el directorio que contiene `batch-report.json`. Un `write` terminado deja ese archivo. `readout` es el comando que lo convierte en el informe.

```text
uv run role-set readout --run out/mock
```

`critic-control` mide al crítico en los elementos revisados. Necesita una subvención y un archivo de latido. `--trait-leg` y `--answer-leg` ejecutan esas dos mediciones.

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` registra los recibos en el almacén. El firewall se ejecuta en la puerta. `store status` informa del archivo. `store void` agrega un valor nulo para una ronda. Los lectores omiten una ronda anulada. No elimina filas.

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` es el lanzador de pago: una copia, luego espera el carril de bucle, escribe y puede volver a lanzarse después de una interrupción por pérdida de conexión. Las pruebas nunca lo llaman contra un punto final activo. CI utiliza `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` ejecuta el solucionador de tipo B sobre un archivo de datos y el `.models.jsonl` adjunto junto a él. Dale todos los archivos de datos del directorio a la vez: un rompecabezas cuyo esqueleto fáctico repite uno anterior, con los nombres, el orden de las líneas y la dirección dejados de lado, se rechaza, indicando el primero.

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## Puertas

Cinco tipos:

- **A.** Un pasaje de enseñanza. Sin veredicto. Se ejecuta un validador, y la verificación de características es consultiva.
- **B.** Un rompecabezas de asignación finita. El solucionador verifica el modelo. La respuesta del crítico es consultiva.
- **C.** Un verificador con un defecto plantado, o un verificador que tenía razón. Los controles de veredicto y evidencia rechazan. El control de defectos se registra y no rechaza.
- **D.** Dos turnos. Los controles de veredicto y evidencia rechazan. La verificación de características es consultiva.
- **E.** Un cuestionario escrito, o un cuestionario de veredicto. Un E escrito no tiene control de veredicto. Un E de tipo veredicto verifica el veredicto y la evidencia.

Los controles de veredicto y evidencia rechazan. Los controles de características, respuesta y defectos se registran y no rechazan, hasta que sus controles se superan. El elemento de características medido representó el 40,3% en general, frente a una barra del 85%, y el peor elemento de características representó el 12,5% frente al 70%. El elemento de respuesta medido representó el 62,7% frente a una barra del 90%. El control de defectos del 11 de octubre de 2026 tuvo 9 de 24. Establecer un control de características o respuesta en esos números rechazaría los elementos que ya se sabe que son buenos.

El solucionador de tipo B cubre rompecabezas de asignación finita escritos en su conjunto cerrado de formas de reglas: como máximo 6 entidades, 3 atributos y 200.000 asignaciones. No verifica el razonamiento con el modelo. Un elemento de tipo B con clave de veredicto sobre material de texto libre está fuera del alcance del solucionador. `puzzle-check` cuenta ese elemento como externo, nunca como superado.

El firewall del almacén rechaza el material retenido, sellado y de entrevista al ingerir, y rechaza el material DEV durante una etapa. No hay ninguna ruta de lectura para el material retenido, sellado o de entrevista. El almacén, la ingestión y el firewall están construidos. La dirección está diseñada y nunca se ha ejecutado.

## Lo que guarda una ejecución

Cada recibo aceptado contiene el hash del plan y el hash del paquete. Cuando la ejecución utilizó ejemplos, el recibo también contiene el commit del ejemplo y el hash del archivo de exclusión. Cuando utilizó la regla estancada, el recibo contiene el hash de esa regla. Una ejecución que no acepta nada no tiene recibos. Cada llamada al escritor se guarda en `drafts.jsonl`, ya sea aceptada o rechazada. Un archivo terminado deja `batch-report.json`, y `readout` informa de ese directorio.

## Modelos

El pin del escritor es Qwen3-235B-A22B-Instruct-2507, la versión AWQ, Apache-2.0. `writers.json` lo sirve como `qwen3-235b-instruct-2507` en la revisión `96548f657d772187f16e9f04bc50aec78a9f9cdd`.

El evaluador es `gemma4:31b`, el digest es `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, el cuantificador es Q4_K_M, Apache-2.0. Su punto final en `writers.json` es `http://127.0.0.1:11434`.

`writers.json` también enumera un escritor que solo realiza pruebas, `qwen3:8b`, y un evaluador de comparación, `granite4.1:30b` (IBM Granite 4.1, Apache-2.0). `write` rechaza el evaluador de comparación como el evaluador de la ejecución. Se rechaza cualquier nombre que no esté en el archivo. Se rechaza cualquier nombre que contenga `cloud`.

Se utiliza la GPU local para el evaluador y para un escritor de prueba, y solo dentro de un proceso. Estos procesos requieren `--grant` y un archivo de "latido" cuyo tiempo de modificación sea inferior a 60 segundos. `critic-control` requiere ambos en cada ejecución. El archivo de "latido" no se abre. Cuando una ejecución carga el evaluador, este se descarga al finalizar, incluso cuando la ejecución se detiene con un error. `--mock` no carga nada.

## Pruebas piloto

Se registran tres pruebas piloto pagadas en [docs/runs/](docs/runs/). Estos son los únicos datos de costos que se indican en este archivo README. Ninguna de sus salidas de generador se incluye en un conjunto de entrenamiento.

- **Prueba piloto 1, 11 de octubre de 2026, 3,52 $.** Ninguna de las 104 filas aceptadas es un dato. El departamento de I+D consideró que las 43 filas C y D estaban mal ensambladas antes de cualquier revisión del contenido. La nota está en [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md).
- **Prueba piloto 2, 11 de octubre de 2026, 3,61 $.** La verificación de claves del departamento de I+D de los 49 candidatos: 15 pasaron la prueba tal como estaban, 15 más necesitaban una corrección mecánica y 19 fallaron. La nota está en [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md).
- **Prueba piloto 3, 11 de octubre de 2026, 2,80 $.** No escribió nada. El motor del escritor se bloqueó después del inicio. La nota está en [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md).

## Modelo de amenazas

role-set es un generador local. No es un servidor. No escucha y no recopila datos de telemetría.

Una ejecución accede al plan, al paquete de roles y al directorio de salida que se especifique. `ingest` y `store` también acceden al archivo de almacenamiento que se especifique. `write`, `pilot-auto` y `critic-control` acceden a un punto final de bucle cuando no son `--mock`. `--mock` no envía nada.

Una ejecución no accede a un servicio de modelo en la nube, a un pod, a una cuenta de alquiler ni a una clave de API de alquiler. No lee filas selladas, conjuntos excluidos ni sesiones de entrevistas. `ingest` sí lee las doce preguntas públicas de la entrevista del clon aspire-si que se especifique, en la confirmación que se especifique, y solo para poder rechazar una fila que las repita. Estas preguntas no se almacenan. `store status` devuelve recuentos, no filas. Informe sobre una vulnerabilidad a través del contacto que se encuentra en [SECURITY.md](SECURITY.md).

## Pruebas

```text
uv sync --frozen
uv run pytest
```

En este árbol hay 241 pruebas y una cobertura del 97,64 %. El mínimo es 90, en `pyproject.toml` y en CI, y la ejecución falla si está por debajo de este valor. CI también ejecuta `role-set plan` y `puzzle-check` en un entorno de prueba, sin modelo. `scripts/verify` ejecuta las mismas comprobaciones locales en un solo comando.

El manual se publica con la [página de inicio](https://mcp-tool-shop-org.github.io/role-set/). Este archivo README es independiente.

Creado por <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
