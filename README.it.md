<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.md">English</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototipo, sviluppato in open source.** A scopo di consulenza e sperimentazione: le interfacce e le strutture dei dati possono cambiare senza preavviso. Nessun elemento definito nell'insieme dei ruoli costituisce un dato di addestramento finché non viene verificato da una persona. Ogni elemento viene sottoposto al controllo chiave da parte del reparto Ricerca e Sviluppo e a una seconda revisione prima di essere incluso in un set di addestramento.

Il piano dell'insieme dei ruoli definisce gli elementi che definiscono il ruolo, scrive le parti che uno scrittore dovrebbe scrivere e registra quali criteri ogni elemento ha superato. Crea anche enigmi di tipo B in codice e li verifica prima che vengano visualizzati da un modello. È un generatore, non un giudice. Ciò che caratterizza un buon ruolo è contenuto in un pacchetto di ruoli, che viene scritto dalle persone.

Lo scrittore e il critico sono raggiungibili solo tramite `http://127.0.0.1` con una porta esplicita. Qualsiasi altra richiesta viene rifiutata prima che venga inviato un byte. L'insieme dei ruoli non noleggia mai una macchina e non tocca mai un pod. Quando un'esecuzione utilizza un pod, Offrig lo noleggia e l'insieme dei ruoli vede solo il tunnel di loopback.

I progetti sono [docs/role-set-design.md](docs/role-set-design.md) e [docs/generator-design.md](docs/generator-design.md), e sono mantenuti in linea con il programma. Le note sull'esecuzione sono disponibili in [docs/runs/](docs/runs/).

## Installazione

Python 3.13 o versione successiva (3.15 è l'obiettivo principale; i test CI utilizzano le versioni 3.13, 3.14 e 3.15). Un'unica dipendenza di runtime: [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 1.7.6, versione fissa. I criteri vengono importati, non copiati.

```text
pip install role-set
```

Per lavorare sull'insieme dei ruoli, clonalo e lascia che uv installi l'ambiente bloccato:

```text
uv sync --frozen
```

## I pacchetti di ruoli

Un pacchetto di ruoli è il testo che le persone hanno scritto per un determinato ruolo. L'insieme dei ruoli lo fornisce in segmenti e non aggiunge nulla. Il suo hash è presente in ogni ricevuta accettata.

- **Il verificatore** (`roles/verifier`). `pack sync` lo copia da un clone locale di aspire-si in un commit: le lezioni, le 18 caratteristiche, la tabella dei tipi, le modifiche e le regole. `pack verify` verifica la copia rispetto al suo manifesto. Un clone aggiornato contiene solo l'elenco delle fonti, l'elenco delle esclusioni e `defects.jsonl`, il catalogo delle classi di difetti rilevate durante le revisioni. La sincronizzazione completa il resto.
- **Il sostenitore** (`roles/advocate`). Un pacchetto scritto da una persona in questo repository, senza testo del modello. Il direttore ha approvato il pacchetto il 14-10-2026 (commit `ffb9546` nella richiesta di pull #56) e il lotto 1 è in fase di creazione; non sarà disponibile su `main` fino al merge di #56. Il lotto 1 ha tre caratteristiche: `failed-gate-first`, `exact-figures` e `faithful-relay`. Nessun elemento del sostenitore costituisce un dato di addestramento finché non supera la lettura di prova.

## Il generatore T10

Gli enigmi di tipo B venivano precedentemente scritti a mano e poi verificati. Ora il codice crea l'enigma e lo verifica, e un modello si limita a formularlo. Il generatore è il pacchetto `role_set.generate`. T10, la lezione sull'uso di parole con dimensioni precise ("esattamente", "almeno", "al massimo"), è la parte che è cambiata nella versione 0.2.0. Viene creata anche una ricetta T12 (quale edizione era in vigore in una data). T11 non viene creato.

**Dimostrato tramite la costruzione.** Il creatore del puzzle propone una soluzione nascosta, aggiunge indizi che sono validi all’interno di tale soluzione fino a quando la risposta non è definitiva, e cancella tutti gli indizi di cui la risposta non ha bisogno. Il risolutore verifica il modello. Quindi, il creatore esegue le quattro letture T10 del reparto Ricerca e Sviluppo: «esattamente» viene interpretato come «almeno», «esattamente» viene interpretato come «al massimo», «almeno» viene interpretato come «esattamente» e «al massimo» viene interpretato come «esattamente». Un puzzle viene mantenuto solo quando una di queste interpretazioni, applicata esclusivamente all’indizio relativo alla caratteristica, modifica la risposta. Un puzzle che supera il test basandosi su un altro tipo di indizio viene classificato come difettoso `trait_not_tested` e viene scartato.

**La mappa delle celle.** Ogni pacchetto, domanda e seme corrisponde a una cella strutturale: il conteggio delle entità `n` (da 3 a 5), la parola di conteggio `op`, `k` (da 1 a n-1) e `ifs`, il numero di indizi `if` (da 0 a 2). Le domande di insieme utilizzano tutte e tre le parole di conteggio. Le domande di affermazione utilizzano solo "esattamente", perché "almeno" e "al massimo" non possono modificare la conclusione di un'affermazione. Le celle utilizzabili vengono sondate una sola volta e registrate in `t10_cells.json`.

**L'asse delle affermazioni `if_shape`.** Una cella di affermazione indica anche la forma dei suoi `if` indizi. Ci sono sette forme: `none` per nessun `if`; `pos` o `neg` per un `if`, in base alla polarità della sua conseguenza; e `chain`, `shared_consequent`, `shared_antecedent` o `disjoint` per due. Una creazione il cui enigma ridotto perde la sua forma riprova all'interno dello stesso seme.

**L'allocatore.** Per una domanda e un conteggio dei valori, un file di 20 semi viene assegnato una sola volta, prima che venga creato qualcosa, tramite un flusso a costo minimo. I 20 semi di ogni pacchetto occupano 20 celle diverse. Una cella senza un indizio `if` viene utilizzata al massimo due volte in un file (il limite zero-if). Le visite successive favoriscono le celle che hanno fornito più scheletri distinti nei semi di ottimizzazione. Questi pesi vengono mantenuti separati per i pacchetti a due valori e a tre valori.

**I limiti di varietà.** In ogni strato (insieme e affermazione, due e tre valori): almeno il 75% di scheletri canonici distinti, nessun scheletro più di 3 volte e nessun scheletro due volte nello stesso file di un pacchetto. I limiti vengono valutati su semi bloccati. Sui semi bloccati da 101 a 120, le affermazioni a tre valori raggiungono 118/120 elementi distinti, con un massimo di 2 ripetizioni.

**Il record registrato.** `t10_record.json` contiene 1.280 creazioni (semi da 1 a 20, da 61 a 80, da 81 a 100 e da 101 a 120, ogni pacchetto, entrambe le domande), ciascuna con la sua cella, `if_shape`, tentativi e scheletro canonico. Contiene un sha256 sulla sorgente del generatore, sulla tabella delle celle, sulla tabella della capacità e sui dati del pacchetto. Un test garantisce che l'elenco dei file sia chiuso rispetto alle importazioni dell'insieme dei ruoli. Il normale lavoro di test CI verifica l'hash e i limiti dal record in pochi minuti. La ricostruzione esaustiva di ogni enigma viene eseguita come lavoro `sweep`, solo su `workflow_dispatch`, e fallisce in caso di qualsiasi differenza rispetto al record.

**Il criterio di capacità e il margine.** `scripts/t10_usable_cells.py --check` segnala la disponibilità, la domanda e il margine per ogni strato e fallisce con i numeri quando un file non può essere posizionato. Con `if_shape`, i pacchetti a tre valori hanno 61 celle di affermazione utilizzabili tra loro, rispetto alle 27 precedenti. Nessun pacchetto necessita di una visita zero-`if` e un test posiziona dodici pacchetti a tre valori (i sei reali e sei cloni) senza ripetizioni all'interno di un file.

## Comandi

`role-set --help` elenca i comandi. Ogni riga qui sotto è un comando che esiste oggi. I flag sono presenti in `--help` di ogni comando.

| Comando | Cosa fa |
|---|---|
| `plan` | Corregge ogni slot per un batch a partire da un seme. Nessuna chiamata al modello. Lo stesso seme genera lo stesso piano. |
| `cost` | Stima le ore di utilizzo dei pod e i costi per un piano prima dell'avvio (`two-card` o `frontier`). Si tratta di una stima. |
| `pack sync` / `pack verify` | Copia il pacchetto del verificatore da un clone locale di aspire-si in un determinato commit e verifica la copia rispetto al suo manifesto. |
| `harvest` | Copia gli estratti con licenza verificata dai repository fissati in un pool locale. La directory del clone non viene registrata. |
| `write` | Il runner. `--mock` lo esegue senza modello e senza costi. Un'esecuzione reale richiede un loopback `--endpoint` e un `--model` da `writers.json`. `--wave write`, `critique` e `rewrite` suddividono un batch in modo che lo scrittore e il critico eseguano un'operazione alla volta, su host diversi se necessario, con `wave-drafts.jsonl` tra di essi. |
| `readout` | Riporta un'esecuzione per ogni lezione e tipo, indicando se è andata a buon fine o meno, dalla directory che contiene `batch-report.json`. |
| `critic-control` | Valuta un critico sugli elementi rivisti, sul suo server llama. Richiede l'hash GGUF e del server del critico, nonché la riga della macchina, oltre a un permesso e un file di heartbeat da un host in affitto. `--trait-leg` e `--answer-leg` eseguono queste fasi. |
| `ingest` / `store` | Registra le ricevute nell'archivio, con il firewall all'ingresso. `store status` riporta i conteggi. `store void` aggiunge una voce di annullamento. |
| `pilot-auto` | Il launcher a pagamento per i piloti del 2026-10-11. Il suo script nomina i pin Qwen e Gemma dei piloti, che non sono presenti in `writers.json`, quindi `write` lo rifiuta. |
| `puzzle-check` | Esegue il risolutore di tipo B sui file di dati e sui relativi `.models.jsonl` file secondari e rifiuta uno scheletro di fatti ripetuto. |

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
| `scripts/t10_cell_capacity.py` | Conta gli scheletri distinti per cella sui semi di ottimizzazione 1, 21 e 41, per strato, in `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Ricostruisce ogni puzzle registrato in `t10_record.json`. `--check` verifica il record creato e stampa le cifre del seme bloccato. |

## Controlli

Cinque tipi:

- **A.** Un brano didattico. Nessuna valutazione. Viene eseguito un validatore e il controllo delle caratteristiche è solo a scopo informativo.
- **B.** Un puzzle di assegnazione finita. Il risolutore verifica il modello. La risposta del critico è solo a scopo informativo.
- **C.** Un controllore con un difetto inserito o un controllore che aveva ragione. I controlli della valutazione e delle prove rifiutano. Il controllo del difetto viene registrato e non rifiuta.
- **D.** Due turni. I controlli della valutazione e delle prove rifiutano. Il controllo delle caratteristiche è solo a scopo informativo.
- **E.** Un quiz scritto o un quiz di valutazione. Un quiz scritto di tipo E non ha un controllo della valutazione. Un quiz di tipo E verifica la valutazione e le prove.

I controlli della valutazione e delle prove rifiutano. I controlli delle caratteristiche, delle risposte e dei difetti vengono registrati e non rifiutano, fino a quando i loro controlli non superano. Il controllo delle caratteristiche ha ottenuto un risultato complessivo del 40,3% rispetto a una soglia dell'85% e la caratteristica peggiore ha ottenuto il 12,5% rispetto al 70%. Il controllo delle risposte ha ottenuto un risultato del 62,7% rispetto a una soglia del 90%. Il controllo dei difetti del 2026-10-11 aveva 9/24.

Il risolutore di tipo B copre i puzzle di assegnazione finita nel suo insieme chiuso di forme di regole: al massimo 6 entità, 3 attributi e 200.000 assegnazioni. Non verifica il ragionamento rispetto al modello. Un elemento di tipo B basato sulla valutazione e su materiale in testo libero è al di fuori del risolutore e `puzzle-check` lo considera tale, non lo considera mai superato.

Il firewall dell'archivio rifiuta il materiale riservato, sigillato e raccolto durante le interviste durante l'inserimento e il materiale DEV durante una fase. Non esiste un percorso di lettura per questo materiale. Lo sterzo è stato progettato e non è mai stato eseguito.

## Cosa conserva un'esecuzione

Ogni ricevuta accettata contiene l'hash del piano e l'hash del pacchetto. Quando l'esecuzione ha utilizzato esempi, contiene anche il commit degli esempi e l'hash del file di esclusione. Quando ha utilizzato la regola bloccata, contiene l'hash di tale regola. La ricevuta di un puzzle generato contiene il suo seme, i tentativi, le letture, le metriche di difficoltà e gli hash dello scheletro, del pacchetto e del risolutore. Ogni chiamata allo scrittore viene conservata in `drafts.jsonl`, accettata o rifiutata.

## Modelli

`writers.json` è l'elenco degli scrittori. Un utilizzo è consentito a meno che una clausola nella licenza o nei termini del creatore non lo limiti; il silenzio non è una limitazione. Ogni voce registra la ricerca dei termini: l'URL della pagina, la data in cui è stata letta, chi l'ha letta, il suo `registry_id` nel registro delle licenze di rnd (`licences/registry.json`) e la sua valutazione di addestramento (`allowed`, `conditions` o `barred`). Una voce senza tale record, o la cui valutazione è `barred`, viene rifiutata quando l'elenco viene caricato. Un nome non presente nell'elenco viene rifiutato, così come un nome che contiene `cloud`.

Gli scrittori sono Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B (questi tre su un pod) e gli otto bracci del pilota: Nemotron 3.5 Bolt 74B (la versione proprietaria di NVIDIA da 24 GB con quantizzazione consapevole), Muse-Glimmer-30B, Nemotron 3.5 Lightning 30B, Qwen3.8-27B, Qwen3.6-35B-A3B, Granite-4.2-30B, Gemma-4-31B-it e DeepSeek-R1-Distill-Qwen-32B. I critici sono Muse-Glimmer-30B e Nemotron 3.5 Lightning. Un critico della stessa famiglia dello scrittore viene rifiutato. Ogni voce fissata contiene un `pin_note`: come è stato creato il suo GGUF e se si adatta a un A100 da 40 GB nel contesto di 32.768 token del pilota.

I modelli sono serviti da llama.cpp's `llama-server`, compilato con CUDA 13.4, all'indirizzo Q8_0. Un modello elencato viene rifiutato finché non viene fissato: la sua revisione del repository, il file GGUF e l'hash del file sha256. Lo script wave esegue l'hash del GGUF e del binario `llama-server` prima dell'avvio e l'esecuzione rifiuta qualsiasi deviazione dall'elenco. Ogni ricevuta registra la partizione (`production` o `trial:<name>`), la famiglia, la revisione e l'hash GGUF di entrambi i modelli, il commit e l'hash del server e la riga della macchina CUDA.

**Valori delle schede.** Lo scrittore e il critico vengono eseguiti con le impostazioni documentate del loro creatore, lette da `experiments/local-builders/card-values.json` con `--cards` (il file J2 viene letto, quindi un modello viene eseguito allo stesso modo in entrambi i casi). Ogni chiave di campionamento e le impostazioni del modello di chat vengono applicate a ogni chiamata, comprese le voci disabilitate, quindi i valori predefiniti di `llama-server` non raggiungono mai un modello. Il server viene eseguito con un contesto di 32.768 token e una risposta può utilizzare 16.384 token, poiché ogni modello con una modalità di ragionamento "pensa". Il testo di sistema di R1-Distill viene integrato nel turno dell'utente, come richiesto dalla sua scheda. Uno scrittore viene associato alla sua scheda tramite il suo ID `card` in `writers.json`, oppure tramite il nome con cui viene identificato. Le ricevute registrano il sha256 del file, il candidato e esattamente ciò che è stato inviato, per lo scrittore e il critico. Un `write` o `critic-control` reale senza `--cards` viene rifiutato.

Un passaggio viene eseguito solo all'interno di un ambiente autorizzato, a meno che la riga della macchina non specifichi una VM di Colab o un pod: `--grant` e un file di heartbeat modificato negli ultimi 60 secondi. Una riga che specifica l'ambiente richiede queste impostazioni, indipendentemente da quanto indicato nella sua scheda. Il file di heartbeat non viene aperto. `--mock` non carica nulla.

## Test preliminari

Tre test preliminari a pagamento sono registrati in [docs/runs/](docs/runs/), tutti del 2026-10-11: $3,52, $3,61 e $2,80. Questi sono gli unici dati sui costi indicati in questo file README. Nessuna riga di questi test è inclusa in un set di addestramento. Nel test 2, il controllo delle chiavi di R&D su 49 candidati ha superato 15 test come scritto, altri 15 dopo una correzione meccanica e ha fallito 19 test. Il test 3 non ha prodotto alcun risultato.

## Modello di minaccia

role-set è un generatore locale. Non è un server. Non ascolta e non raccoglie dati di telemetria.

Un'esecuzione influisce sul piano, sul pacchetto di ruoli e sulla directory di output specificata. `ingest` e `store` influiscono anche sul file di archiviazione specificato. `write`, `pilot-auto` e `critic-control` influiscono su un endpoint di loopback quando non sono `--mock`. `--mock` non invia nulla. Il generatore e i suoi script leggono e scrivono file all'interno del repository e non chiamano alcun modello.

Un'esecuzione non influisce su un servizio di modello cloud, un pod, un account a noleggio o una chiave API a noleggio. Non legge righe sigillate, set esclusi o sessioni di intervista. `ingest` legge le dodici domande pubbliche dell'intervista dal clone di aspire-si specificato, alla versione specificata, solo per rifiutare una riga che le ripete. Queste non vengono archiviate. `store status` restituisce conteggi, non righe. Segnala una vulnerabilità tramite il contatto in [SECURITY.md](SECURITY.md).

## Test

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La copertura delle righe ha un valore minimo del 90%, in `pyproject.toml` e `codecov.yml`, e l'esecuzione fallisce se questo valore non viene raggiunto. Misura circa il 97%. Il badge di copertura mostrato sopra indica il valore corrente. CI esegue anche il controllo di atlas, `role-set plan` e `puzzle-check` su un elemento di test, senza utilizzare alcun modello. `scripts/verify` esegue i controlli locali in un unico comando.

## Collaboratori

Alcune parti di questo repository sono state scritte con Grok (xAI). Grok, Argon e Claude sono menzionati in [CONTRIBUTORS.md](CONTRIBUTORS.md).

Il manuale viene pubblicato con la [pagina di destinazione](https://mcp-tool-shop-org.github.io/role-set/). Questo file README è autonomo.

Creato da <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
