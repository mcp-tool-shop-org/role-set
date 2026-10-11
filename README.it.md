<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototipo, sviluppato in modalità open source.** Consulenza e sperimentazione: le interfacce e le strutture dei dati cambiano senza preavviso.

Il set di ruoli definisce gli elementi che modellano il ruolo, scrive le parti che uno scrittore dovrebbe scrivere e registra quali "gate" (controlli) ha superato ciascun elemento. È un generatore, non un giudice. Le caratteristiche di un buon verificatore risiedono nel pacchetto di ruoli, che viene scritto dagli esseri umani. Nulla di ciò che questo programma scrive è incluso in un set di addestramento. Ogni elemento passa attraverso il controllo chiave di R&D e una seconda revisione prima di avvicinarsi a uno di essi.

Lo scrittore e il critico sono raggiungibili solo tramite `http://127.0.0.1` con una porta esplicita. Qualsiasi altra richiesta viene rifiutata prima che venga inviato un singolo byte. Il set di ruoli non noleggia mai una macchina e non interagisce mai con un "pod". Quando un'esecuzione utilizza un pod, è "offrig" che lo noleggia e il set di ruoli vede solo il tunnel di loopback.

Il progetto è descritto nel documento [docs/role-set-design.md](docs/role-set-design.md) e rispetta le linee guida del programma. Le note relative all’esecuzione sono disponibili in [docs/runs/](docs/runs/).

## Installazione

Python 3.12, gestito da uv. Un'unica dipendenza di runtime: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si), fissata a un commit di aspire-si. I "gate" vengono importati, non copiati.

```text
uv sync --frozen
```

## Comandi

`role-set --help` elenca i comandi. Ogni riga qui sotto è un comando che esiste attualmente.

`plan` definisce ogni "slot" per un batch a partire da un "seed". Non chiama un modello. Lo stesso "seed" riproduce il piano.

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` stima le ore e i costi del pod per quel piano prima che venga avviato qualsiasi processo. La cifra è una stima. I profili sono `two-card` e `frontier`.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` copia il pacchetto di ruoli da un clone locale di aspire-si a un commit specifico. `pack verify` verifica tale copia rispetto al suo "manifest". Il percorso del clone non viene registrato. Un clone aggiornato di questo repository non è un pacchetto sincronizzato: `roles/verifier` contiene l'elenco delle esclusioni e l'elenco delle fonti, e la sincronizzazione completa il resto.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` copia gli estratti con licenza verificata in un pool locale. La directory del clone non viene registrata.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` è l'esecutore. `--mock` esegue l'intera pipeline senza un modello e senza costi, e non richiede un finanziamento. L'output di simulazione è contrassegnato, non è un dato e viene scritto in `<out>/mock/`. Un'esecuzione che chiama un modello richiede `--endpoint` e `--model`, e l'endpoint deve essere di loopback. Il caricamento del critico, o di uno scrittore di prove locale, richiede anche `--grant` e `--heartbeat`. `--no-critic` è un'esecuzione di prova: gli elementi passano senza il secondo lettore e lo scrittore del pod non utilizza quindi la GPU locale.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` genera un report per ogni esecuzione per lezione e tipo, inclusa un'esecuzione che non ha accettato nulla. Legge la directory che contiene `batch-report.json`. Un `write` completato lascia quel file. `readout` è il comando che lo trasforma nel report.

```text
uv run role-set readout --run out/mock
```

`critic-control` misura il critico sugli elementi rivisti. Richiede un finanziamento e un file "heartbeat". `--trait-leg` e `--answer-leg` eseguono queste due misurazioni.

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` registra le ricevute nello storage. Il firewall è attivo all'ingresso. `store status` genera il report del file. `store void` aggiunge un "void" per un round. I lettori saltano un round "void". Non elimina le righe.

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` è il lanciatore a pagamento: una singola richiesta, quindi attende il canale di loopback, scrive e può essere riavviato dopo un'interruzione dovuta a una perdita di connessione. I test non lo chiamano mai su un endpoint attivo. CI utilizza `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` esegue il risolutore di tipo B su un file di dati e il "sidecar" `.models.jsonl` accanto ad esso. Fornisci tutti i file di dati nella directory contemporaneamente: un puzzle la cui struttura fattuale ripete una precedente, con nomi, ordine delle righe e direzione messi da parte, viene rifiutato, indicando il primo.

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## "Gate" (Controlli)

Cinque tipi:

- **A.** Un passaggio didattico. Nessun giudizio. Viene eseguito un validatore e il controllo delle caratteristiche è solo a scopo di consulenza.
- **B.** Un puzzle di assegnazione finito. Il risolutore verifica il modello. La risposta del critico è solo a scopo di consulenza.
- **C.** Un controllore con un difetto inserito, o un controllore che aveva ragione. I controlli del giudizio e delle prove rifiutano. Il controllo del difetto viene registrato e non rifiuta.
- **D.** Due turni. I controlli del giudizio e delle prove rifiutano. Il controllo delle caratteristiche è solo a scopo di consulenza.
- **E.** Un quiz scritto o un quiz di giudizio. Un E scritto non ha un controllo del giudizio. Un E di tipo giudizio verifica il giudizio e le prove.

I controlli del giudizio e delle prove rifiutano. I controlli delle caratteristiche, delle risposte e dei difetti vengono registrati e non rifiutano, fino a quando i loro controlli non vengono superati. Il controllo delle caratteristiche ha ottenuto una media del 40,3% rispetto a una soglia dell'85% e la caratteristica peggiore ha ottenuto il 12,5% rispetto al 70%. Il controllo delle risposte ha ottenuto una media del 62,7% rispetto a una soglia del 90%. Il controllo dei difetti dell'11-10-2026 aveva 9 difetti su 24. L'applicazione del controllo delle caratteristiche o delle risposte a tali numeri rifiuterebbe elementi già noti come validi.

Il risolutore di tipo B copre i puzzle di assegnazione finita scritti nel suo set chiuso di forme di regole: al massimo 6 entità, 3 attributi e 200.000 assegnazioni. Non verifica il ragionamento rispetto al modello. Un elemento di tipo B basato su giudizi su materiale in testo libero è al di fuori del campo del risolutore. `puzzle-check` conta tale elemento come esterno, mai come superato.

Il firewall dello storage rifiuta il materiale "held-out", sigillato e delle interviste durante l'inserimento e rifiuta il materiale DEV durante una fase. Non esiste un percorso di lettura per il materiale "held-out", sigillato o delle interviste. Lo storage, l'inserimento e il firewall sono stati creati. Il controllo è stato progettato e non è mai stato eseguito.

## Cosa conserva un'esecuzione

Ogni ricevuta accettata contiene l'hash del piano e l'hash del pacchetto. Quando l'esecuzione ha utilizzato degli esempi, la ricevuta contiene anche il commit dell'esempio e l'hash del file di esclusione. Quando ha utilizzato la regola "stuck", la ricevuta contiene l'hash di tale regola. Un'esecuzione che non accetta nulla non ha ricevute. Ogni chiamata allo scrittore viene conservata in `drafts.jsonl`, accettata o rifiutata. Un'operazione di scrittura completata lascia `batch-report.json` e `readout` genera il report di tale directory.

## Modelli

Il "pin" dello scrittore è Qwen3-235B-A22B-Instruct-2507, la versione AWQ, Apache-2.0. `writers.json` lo serve come `qwen3-235b-instruct-2507` alla revisione `96548f657d772187f16e9f04bc50aec78a9f9cdd`.

Il critico è `gemma4:31b`, il digest è `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, il quant è Q4_K_M, licenza Apache-2.0. Il suo endpoint in `writers.json` è `http://127.0.0.1:11434`.

`writers.json` elenca anche uno strumento di scrittura utilizzato solo per le prove, `qwen3:8b`, e uno strumento di confronto, `granite4.1:30b` (IBM Granite 4.1, Apache-2.0). `write` rifiuta lo strumento di confronto come strumento critico per l'esecuzione. Un nome che non è presente nel file viene rifiutato. Un nome che contiene `cloud` viene rifiutato.

La GPU locale viene utilizzata per lo strumento critico e per lo strumento di scrittura utilizzato per le prove, e solo all'interno di un processo. Questi percorsi richiedono `--grant` e un file di "heartbeat" il cui tempo di modifica è inferiore a 60 secondi. `critic-control` richiede entrambi ad ogni esecuzione. Il file di "heartbeat" non viene aperto. Quando un'esecuzione carica lo strumento critico, questo viene scaricato al termine, anche quando l'esecuzione si interrompe a causa di un errore. `--mock` non carica nulla.

## Esecuzioni di prova

Tre esecuzioni di prova a pagamento sono registrate in [docs/runs/](docs/runs/). Queste sono le uniche cifre di costo indicate in questo file README. Nessun output del generatore proveniente da queste esecuzioni è incluso in un set di addestramento.

- **Esecuzione di prova 1, 2026-10-11, $3.52.** Nessuna delle 104 righe accettate contiene dati. Il reparto R&D ha contrassegnato le 43 righe C e D come assemblate in modo errato prima di qualsiasi revisione del contenuto. La nota è disponibile all'indirizzo [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md).
- **Esecuzione di prova 2, 2026-10-11, $3.61.** Il reparto R&D ha eseguito un controllo sui 49 candidati: 15 sono stati approvati come presentati, altri 15 necessitavano di una correzione meccanica e 19 sono stati rifiutati. La nota è disponibile all'indirizzo [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md).
- **Esecuzione di prova 3, 2026-10-11, $2.80.** Non ha prodotto alcun output. Il motore dello strumento di scrittura si è bloccato dopo l'avvio. La nota è disponibile all'indirizzo [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md).

## Modello di minaccia

"role-set" è un generatore locale. Non è un server. Non ascolta e non raccoglie dati di telemetria.

Un'esecuzione accede al piano, al pacchetto di ruoli e alla directory di output specificata. `ingest` e `store` accedono anche al file di archivio specificato. `write`, `pilot-auto` e `critic-control` accedono a un endpoint di loopback quando non sono `--mock`. `--mock` non invia nulla.

Un'esecuzione non accede a un servizio di modello cloud, a un pod, a un account di noleggio o a una chiave API di noleggio. Non legge righe sigillate, set esclusi o sessioni di intervista. `ingest` legge le dodici domande pubbliche dell'intervista dal clone di "aspire-si" specificato, alla versione specificata, e solo per poter rifiutare una riga che le ripete. Queste domande non vengono archiviate. `store status` restituisce conteggi, non righe. Segnala una vulnerabilità tramite il contatto presente in [SECURITY.md](SECURITY.md).

## Test

```text
uv sync --frozen
uv run pytest
```

In questo albero sono presenti 241 test e una copertura del 97,64%. Il valore minimo è 90, in `pyproject.toml` e in CI, e l'esecuzione fallisce se questo valore non viene raggiunto. CI esegue anche `role-set plan` e `puzzle-check` su un elemento di test, senza modello. `scripts/verify` esegue gli stessi controlli locali in un unico comando.

Il manuale è pubblicato insieme alla [pagina di destinazione](https://mcp-tool-shop-org.github.io/role-set/). Questo file README è autonomo.

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
