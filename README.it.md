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

**Prototipo, sviluppato in modalità open source.** A scopo di consulenza e sperimentazione: le interfacce e le strutture dei dati possono cambiare senza preavviso. Nessun elemento definito nell'insieme di ruoli costituisce un dato di addestramento finché non viene verificato da una persona. Ogni elemento viene sottoposto al controllo chiave da parte del reparto Ricerca e Sviluppo e a una seconda revisione prima di essere incluso in un set di addestramento.

L'insieme di ruoli pianifica gli elementi che definiscono il ruolo, scrive le parti che uno scrittore dovrebbe scrivere e registra quali criteri ogni elemento ha superato. Crea anche enigmi di tipo B in codice e li verifica prima che vengano visualizzati da un modello. È un generatore, non un giudice. Ciò che caratterizza un buon ruolo è contenuto in un pacchetto di ruoli, che viene scritto dalle persone.

Lo scrittore e il critico possono essere contattati solo tramite `http://127.0.0.1`, utilizzando una porta esplicita. Qualsiasi altra richiesta viene rifiutata prima dell'invio di un singolo byte. L'insieme di ruoli non noleggia mai una macchina e non accede mai a un pod. Quando un'esecuzione utilizza un pod, Offrig lo noleggia e l'insieme di ruoli visualizza solo il tunnel di loopback.

Le specifiche sono disponibili qui: [docs/role-set-design.md](docs/role-set-design.md) e [docs/generator-design.md](docs/generator-design.md), e sono mantenute in linea con il programma. Le note sull'esecuzione sono disponibili in [docs/runs/](docs/runs/).

## Installazione

Python 3.13+ (3.15), gestito da uv. Un'unica dipendenza di runtime: [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si) 1.4.0, fissata alla commit aspire-si `388a6bc`. Le porte vengono importate, non copiate.

```text
uv sync --frozen
```

## I pacchetti di ruoli

Un pacchetto di ruoli è il testo che le persone hanno scritto per un determinato ruolo. L'insieme di ruoli lo fornisce in segmenti e non aggiunge nulla. Il suo hash è presente in ogni ricevuta accettata.

- **Il verificatore** (`roles/verifier`). `pack sync` lo copia da un clone locale di aspire-si in una commit specifica: le lezioni, le 18 caratteristiche, la tabella dei tipi, le modifiche e le regole. `pack verify` verifica la copia rispetto al suo manifesto. Un clone aggiornato contiene solo l'elenco delle fonti, l'elenco delle esclusioni e `defects.jsonl`, il catalogo delle classi di difetti rilevate durante le revisioni. La sincronizzazione completa il resto.
- **Il sostenitore** (`roles/advocate`). Un pacchetto scritto da una persona in questo repository, senza testo di modello al suo interno. Il direttore ha approvato il pacchetto il 14 ottobre 2026 (commit `ffb9546` sulla richiesta di pull n. 56) e il lotto 1 è in fase di creazione; non sarà disponibile su `main` fino al completamento della richiesta di pull n. 56. Il lotto 1 ha tre caratteristiche: `failed-gate-first`, `exact-figures` e `faithful-relay`. Nessun elemento del sostenitore costituisce un dato di addestramento finché non supera la fase di verifica preliminare.

## Il generatore T10

Gli enigmi di tipo B venivano precedentemente scritti a mano e poi verificati. Ora, il codice crea l'enigma e lo verifica, e un modello si limita a formularlo. Il generatore è il pacchetto `role_set.generate`. T10, la lezione sull'uso di termini che indicano dimensioni precise ("esattamente", "almeno", "al massimo"), è la parte che è stata modificata nella versione 0.2.0. Viene anche creato un ricettario T12 (che indica quale versione era in vigore in una determinata data). T11 non viene creato.

**Verificato tramite la costruzione.** Il creatore campiona una soluzione nascosta, aggiunge indizi che sono veri al suo interno finché la risposta non è definita e cancella ogni indizio di cui la risposta non ha bisogno. Il risolutore verifica il modello. Quindi, il creatore esegue le quattro letture T10 del reparto Ricerca e Sviluppo: "esattamente" letto come "almeno", "esattamente" letto come "al massimo", "almeno" letto come "esattamente" e "al massimo" letto come "esattamente". Un enigma viene conservato solo quando una di queste letture, applicata all'indizio della caratteristica, modifica la risposta. Un enigma che supera il test con un altro tipo di indizio è classificato come `trait_not_tested` e viene rifiutato.

**La mappa delle celle.** Ogni pacchetto, domanda e seme corrisponde a una cella strutturale: il numero di entità `n` (da 3 a 5), la parola che indica la quantità `op`, `k` (da 1 a n-1) e `ifs`, il numero di indizi `if` (da 0 a 2). Le domande di tipo "set" utilizzano tutte e tre le parole che indicano la quantità. Le domande di tipo "affermazione" utilizzano solo "esattamente", perché "almeno" e "al massimo" non possono modificare la validità di un'affermazione. Le celle utilizzabili vengono esaminate una sola volta e vengono memorizzate in `t10_cells.json`.

**L'asse delle affermazioni `if_shape`.** Una cella di affermazione indica anche la forma dei suoi `if` indizi. Esistono sette forme: `none` per nessun `if`; `pos` o `neg` per un singolo `if`, in base alla polarità della sua conseguenza; e `chain`, `shared_consequent`, `shared_antecedent` o `disjoint` per due. Una creazione il cui enigma ridotto perde la sua forma riprova all'interno dello stesso seme.

**L'allocatore.** Per una domanda e un conteggio dei valori, un file di 20 semi viene assegnato una sola volta, prima che venga creata qualsiasi cosa, tramite un flusso a costo minimo. I 20 semi di ogni pacchetto occupano 20 celle diverse. Una cella che non contiene alcun indizio `if` viene utilizzata al massimo due volte in un file (il limite zero-if). Le visite successive favoriscono le celle che hanno fornito più schemi distinti durante i test. Questi pesi vengono mantenuti separati per i pacchetti a due e tre valori.

**I limiti di varietà.** In ogni strato (set e affermazione, due e tre valori): almeno il 75% di schemi canonici distinti, nessun schema ripetuto più di 3 volte e nessun schema ripetuto due volte nello stesso file di un pacchetto. I limiti vengono valutati sui semi bloccati. Sui semi bloccati da 101 a 120, le affermazioni a tre valori raggiungono 118 su 120 distinti, con una ripetizione massima di 2.

**La registrazione memorizzata.** `t10_record.json` contiene 1.280 creazioni (semi da 1 a 20, da 61 a 80, da 81 a 100 e da 101 a 120, ogni pacchetto, entrambe le domande), ciascuna con la sua cella, `if_shape`, tentativi e schema canonico. Contiene un hash SHA256 sul codice sorgente del generatore, sulla tabella delle celle, sulla tabella della capacità e sui dati del pacchetto. Un test verifica che l'elenco dei file sia chiuso rispetto alle importazioni dell'insieme di ruoli. Il normale processo di test CI verifica l'hash e i limiti dalla registrazione in pochi minuti. La ricostruzione completa di ogni enigma viene eseguita come processo `sweep`, solo su `workflow_dispatch`, e fallisce in caso di qualsiasi differenza rispetto alla registrazione.

**La porta della capacità e lo spazio libero.** `scripts/t10_usable_cells.py --check` segnala la disponibilità, la domanda e lo spazio libero per ogni strato e fallisce se un file non può essere inserito. Con `if_shape`, i pacchetti a tre valori hanno 61 celle di affermazione utilizzabili, rispetto alle 27 precedenti. Nessun pacchetto richiede una visita zero-`if` e un test inserisce dodici pacchetti a tre valori (i sei reali e sei cloni) senza ripetizioni all'interno di un file.

## Comandi

`role-set --help` elenca i comandi. Ogni riga qui sotto è un comando che esiste oggi. I flag sono presenti nella `--help` di ogni comando.

| Comando | Cosa fa |
|---|---|
| `plan` | Definisce ogni slot per un lotto a partire da un seme. Non richiede l'intervento del modello. Lo stesso seme genera lo stesso piano. |
| `cost` | Stima le ore di lavoro e i costi per un piano prima dell'avvio (`two-card` o `frontier`). Si tratta di una stima. |
| `pack sync` / `pack verify` | Copia il pacchetto del verificatore da una copia locale di aspire-si in un determinato commit e verifica la copia rispetto al suo manifesto. |
| `harvest` | Copia gli estratti con licenza verificata dai repository selezionati in un pool locale. La directory della copia non viene registrata. |
| `write` | Il runner. `--mock` lo esegue senza modello e senza costi. Un'esecuzione reale richiede un loopback `--endpoint` e un `--model` da `writers.json`. |
| `readout` | Riporta un'esecuzione per ogni lezione e tipo, indicando se è andata a buon fine o meno, dalla directory che contiene `batch-report.json`. |
| `critic-control` | Misura il valutatore sugli elementi revisionati. Richiede un finanziamento e un file di heartbeat. `--trait-leg` e `--answer-leg` eseguono queste fasi. |
| `ingest` / `store` | Registra le ricevute nel database, con il firewall all'ingresso. `store status` riporta i conteggi. `store void` aggiunge una voce di annullamento. |
| `pilot-auto` | Il launcher a pagamento: attende il loopback, scrive e può essere riavviato dopo un'interruzione dovuta a una chiamata persa. |
| `puzzle-check` | Esegue il risolutore di tipo B sui file di dati e sui relativi `.models.jsonl` file di supporto e rifiuta uno schema di fatti ripetuto. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

Il generatore non ha ancora il comando `role-set`. I suoi tre script creano le tabelle e i record:

| Script | Cosa fa |
|---|---|
| `scripts/t10_usable_cells.py` | Verifica ogni cella e scrive `t10_cells.json`. `--check` esegue il controllo della capacità sulla tabella creata e non esegue la verifica. |
| `scripts/t10_cell_capacity.py` | Conta gli schemi distinti per cella sui set di ottimizzazione 1, 21 e 41, per livello, in `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Ricostruisce ogni puzzle registrato in `t10_record.json`. `--check` verifica il record creato e stampa le cifre del seme bloccato. |

## Controlli

Cinque tipi:

- **A.** Un brano didattico. Nessuna valutazione. Viene eseguito un validatore e il controllo delle caratteristiche è solo a scopo informativo.
- **B.** Un puzzle di assegnazione finita. Il risolutore verifica il modello. La risposta del valutatore è solo a scopo informativo.
- **C.** Un controllore con un difetto inserito o un controllore che aveva ragione. I controlli sulla valutazione e sull'evidenza vengono rifiutati. Il controllo del difetto viene registrato e non viene rifiutato.
- **D.** Due turni. I controlli sulla valutazione e sull'evidenza vengono rifiutati. Il controllo delle caratteristiche è solo a scopo informativo.
- **E.** Un quiz scritto o un quiz sulla valutazione. Un quiz scritto di tipo E non ha un controllo sulla valutazione. Un quiz di tipo E sulla valutazione verifica la valutazione e l'evidenza.

I controlli sulla valutazione e sull'evidenza vengono rifiutati. I controlli sulle caratteristiche, sulla risposta e sui difetti vengono registrati e non vengono rifiutati, fino a quando i loro controlli non superano il test. Il controllo delle caratteristiche ha misurato una media del 40,3% rispetto a una soglia dell'85% e la caratteristica peggiore ha ottenuto il 12,5% rispetto al 70%. Il controllo della risposta ha misurato il 62,7% rispetto a una soglia del 90%. Il controllo del difetto del 2026-10-11 ha ottenuto 9 risultati positivi su 24.

Il risolutore di tipo B gestisce i puzzle di assegnazione finita nel suo set chiuso di forme di regola: al massimo 6 entità, 3 attributi e 200.000 assegnazioni. Non verifica il ragionamento rispetto al modello. Un elemento di tipo B basato sulla valutazione su materiale in testo libero è al di fuori del risolutore e `puzzle-check` lo considera tale, non lo considera mai superato.

Il firewall del database rifiuta il materiale riservato, sigillato e quello raccolto durante le interviste durante l'inserimento e il materiale DEV durante una fase. Non esiste un percorso di lettura per questo materiale. Il sistema di controllo è stato progettato ma non è mai stato eseguito.

## Cosa conserva un'esecuzione

Ogni ricevuta accettata contiene l'hash del piano e l'hash del pacchetto. Quando l'esecuzione ha utilizzato esempi, contiene anche il commit dell'esempio e l'hash del file di esclusione. Quando ha utilizzato la regola bloccata, contiene l'hash di tale regola. La ricevuta di un puzzle generato contiene il suo seme, i tentativi, le letture, le metriche di difficoltà e gli hash dello schema, del pacchetto e del risolutore. Ogni chiamata di scrittura viene conservata in `drafts.jsonl`, sia che venga accettata o rifiutata.

## Modelli

Il pin dello scrittore è Qwen3-235B-A22B-Instruct-2507, la build AWQ, Apache-2.0, utilizzato come `qwen3-235b-instruct-2507` nella revisione `96548f657d772187f16e9f04bc50aec78a9f9cdd`. Il pin del valutatore è `gemma4:31b`, digest `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, Q4_K_M, Apache-2.0, in `http://127.0.0.1:11434`. `writers.json` elenca anche uno scrittore solo per le prove e un valutatore di confronto. Un nome non presente nel file viene rifiutato, così come un nome che contiene `cloud`.

La GPU locale viene utilizzata per il valutatore e per uno scrittore di prova, solo all'interno di un finanziamento: `--grant` e un file di heartbeat modificato negli ultimi 60 secondi. Il file di heartbeat non viene aperto. Un valutatore caricato durante l'esecuzione viene scaricato al termine, anche in caso di errore. `--mock` non carica nulla.

## Test pilota

Tre test pilota a pagamento sono registrati in [docs/runs/](docs/runs/), tutti del 2026-10-11: $3,52, $3,61 e $2,80. Queste sono le uniche cifre di costo indicate in questo file README. Nessuna riga di questi test è presente in un set di addestramento. Nel test pilota 2, il controllo delle chiavi di R&D su 49 candidati ha approvato 15 candidati come scritto, altri 15 dopo una correzione meccanica e ha rifiutato 19. Il test pilota 3 non ha scritto nulla.

## Modello di minaccia

role-set è un generatore locale. Non è un server. Non ascolta e non raccoglie dati di telemetria.

Un'esecuzione tocca il piano, il pacchetto di ruolo e la directory di output specificata. `ingest` e `store` toccano anche il file del database specificato. `write`, `pilot-auto` e `critic-control` toccano un endpoint di loopback quando non sono `--mock`. `--mock` non invia nulla. Il generatore e i suoi script leggono e scrivono file all'interno del repository e non chiamano alcun modello.

Un'esecuzione non tocca un servizio di modello cloud, un pod, un account di noleggio o una chiave API di noleggio. Non legge righe sigillate, set riservati o registrazioni di interviste. `ingest` legge le dodici domande pubbliche delle interviste dalla copia di aspire-si specificata, nel commit specificato, solo per rifiutare una riga che le ripete. Queste non vengono archiviate. `store status` restituisce conteggi, non righe. Segnala una vulnerabilità tramite il contatto in [SECURITY.md](SECURITY.md).

## Test

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La copertura delle righe ha una soglia minima del 90%, in `pyproject.toml` e `codecov.yml`, e l'esecuzione fallisce se non la raggiunge. Attualmente, la copertura è di circa il 97%. Il badge di copertura sopra mostra la cifra attuale. Il CI esegue anche il controllo di atlas, `role-set plan` e `puzzle-check` su un elemento di test, senza modello. `scripts/verify` esegue i controlli locali in un unico comando.

Il manuale è pubblicato con la [pagina di destinazione](https://mcp-tool-shop-org.github.io/role-set/). Questo file README è autonomo.

Realizzato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>.
