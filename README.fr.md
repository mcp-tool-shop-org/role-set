<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototype, développé en mode ouvert.** Conseils et expérimentation : les interfaces et les structures de données sont modifiées sans préavis. Rien de ce qui est écrit dans l’ensemble des rôles n’est considéré comme des données d’entraînement tant qu’une personne ne l’a pas vérifié. Chaque élément est soumis à la vérification clé du département R&D et à une deuxième lecture avant de se rapprocher d’un ensemble d’entraînement.

L’ensemble des rôles planifie les éléments de définition des rôles, écrit les parties qu’un rédacteur doit écrire et enregistre les étapes que chaque élément a franchies. Il crée également des énigmes de type B en code et les valide avant qu’un modèle ne les voie. Il s’agit d’un générateur, et non d’un juge. Ce qui définit un bon rôle se trouve dans un ensemble de rôles, que les personnes rédigent.

Le rédacteur et le critique ne sont accessibles qu’à l’adresse `http://127.0.0.1`, via un port explicite. Tout autre accès est refusé avant l’envoi d’un octet. L’ensemble des rôles ne loue jamais de machine et ne touche jamais à un pod. Lorsqu’une exécution utilise un pod, offrig le loue, et l’ensemble des rôles ne voit que le tunnel de boucle.

Les conceptions sont disponibles à l’adresse [docs/role-set-design.md](docs/role-set-design.md) et [docs/generator-design.md](docs/generator-design.md), et sont maintenues en cohérence avec le programme. Les notes d’exécution se trouvent dans [docs/runs/](docs/runs/).

## Installation

Python 3.12, géré par uv. Une seule dépendance d’exécution : [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si) 1.4.0, fixé à la validation aspire-si `388a6bc`. Les étapes sont importées, et non copiées.

```text
uv sync --frozen
```

## Les ensembles de rôles

Un ensemble de rôles est le texte que les personnes ont rédigé pour un rôle. L’ensemble des rôles le fournit en tranches et n’y ajoute rien. Son hachage figure sur chaque reçu accepté.

- **Le vérificateur** (`roles/verifier`). `pack sync` le copie à partir d’un clone local d’aspire-si à une validation : les leçons, les 18 caractéristiques, le tableau des types, les amendements et les règles. `pack verify` vérifie la copie par rapport à son manifeste. Un clone récent ne contient que la liste des sources, la liste d’exclusion et `defects.jsonl`, le catalogue des classes de défauts trouvées lors des révisions. La synchronisation complète le reste.
- **Le défenseur** (`roles/advocate`). Un ensemble qu’une personne a rédigé dans ce dépôt, sans texte de modèle. Le directeur a approuvé l’ensemble le 2026-10-14 (validation `ffb9546` sur la demande d’extraction n° 56), et le lot 1 est en cours de construction ; il ne se trouve pas sur `main` tant que la demande d’extraction n° 56 n’est pas fusionnée. Le lot 1 comporte trois caractéristiques : `failed-gate-first`, `exact-figures` et `faithful-relay`. Aucun élément du défenseur n’est considéré comme des données d’entraînement tant qu’il n’a pas passé la lecture pilote.

## Le générateur T10

Les énigmes de type B étaient auparavant rédigées à la main, puis vérifiées. Désormais, le code crée l’énigme et la valide, et un modèle se contente de la formuler. Le générateur est le paquet `role_set.generate`. T10, la leçon sur les mots de taille stricte (« exactement », « au moins », « au plus »), est la partie qui a changé dans la version 0.2.0. Une recette T12 (quelle édition était en vigueur à une date donnée) est également créée. T11 n’est pas créé.

**Validé par construction.** Le créateur échantillonne une solution cachée, ajoute des indices qui sont vrais dans cette solution jusqu’à ce que la réponse soit définie, et supprime tous les indices dont la réponse n’a pas besoin. Le résolveur vérifie le modèle. Ensuite, le créateur exécute les quatre lectures T10 du département R&D : « exactement » lu comme « au moins », « exactement » lu comme « au plus », « au moins » lu comme « exactement » et « au plus » lu comme « exactement ». Une énigme est conservée uniquement si l’une de ces lectures, appliquée seule à l’indice de caractéristique, modifie la réponse. Une énigme qui réussit sur un autre indice de comptage est la classe de défaut `trait_not_tested`, et elle est rejetée.

**La carte des cellules.** Chaque ensemble, question et valeur de départ est mappé à une cellule structurelle : le nombre d’entités `n` (de 3 à 5), le mot de comptage `op`, `k` (de 1 à n-1) et `ifs`, le nombre d’indices `if` (de 0 à 2). Les questions d’ensemble utilisent les trois mots de comptage. Les questions d’affirmation n’utilisent que « exactement », car « au moins » et « au plus » ne peuvent pas modifier le verdict d’une affirmation. Les cellules utilisables sont sondées une fois et validées dans `t10_cells.json`.

**L’axe d’affirmation `if_shape`.** Une cellule d’affirmation donne également le nom de la forme de ses `if` indices. Il existe sept formes : `none` pour aucun `if` ; `pos` ou `neg` pour un `if`, en fonction de la polarité de sa conséquence ; et `chain`, `shared_consequent`, `shared_antecedent` ou `disjoint` pour deux. Une construction dont l’énigme réduite perd sa forme tente de nouveau dans la même valeur de départ.

**L’allocateur.** Pour une question et un nombre de valeurs, un fichier de 20 valeurs de départ est attribué une fois, avant que quoi que ce soit ne soit construit, par un flux de coût minimal. Les 20 valeurs de départ de chaque ensemble prennent 20 cellules différentes. Une cellule qui ne contient pas d’indice `if` est utilisée au plus deux fois dans un fichier (la limite zéro-si). Les visites suivantes privilégient les cellules qui ont donné plus de squelettes distincts lors des valeurs de départ de réglage. Ces pondérations sont conservées séparément pour les ensembles à deux et trois valeurs.

**Les seuils de variété.** Dans chaque strate (ensemble et affirmation, deux et trois valeurs) : au moins 75 % de squelettes canoniques distincts, aucun squelette n’apparaît plus de trois fois et aucun squelette n’apparaît deux fois dans le même fichier d’un ensemble. Les seuils sont évalués sur les valeurs de départ verrouillées. Sur les valeurs de départ verrouillées 101 à 120, les affirmations à trois valeurs atteignent 118 sur 120 valeurs distinctes, avec une répétition maximale de 2.

**L’enregistrement validé.** `t10_record.json` contient 1 280 constructions (valeurs de départ 1 à 20, 61 à 80, 81 à 100 et 101 à 120, chaque ensemble, les deux questions), chacune avec sa cellule, `if_shape`, ses tentatives et son squelette canonique. Il contient un hachage sha256 sur la source du générateur, le tableau des cellules, le tableau des capacités et les données de l’ensemble. Un test garantit que cette liste de fichiers est fermée par les importations de l’ensemble des rôles. Le test CI normal vérifie le hachage et les seuils à partir de l’enregistrement en quelques minutes. La reconstruction exhaustive de chaque énigme s’exécute en tant que tâche `sweep`, uniquement sur `workflow_dispatch`, et échoue en cas de différence par rapport à l’enregistrement.

**The capacity gate and the headroom.** `scripts/t10_usable_cells.py --check` reports supply, demand and headroom for each stratum, and fails with the numbers when a file cannot be placed. With `if_shape`, three-value packs have 61 usable claim cells between them, against 27 before. No pack needs a zero-`if` visit, and a test places twelve three-value packs (the six real ones and six clones) with no repeat inside a file.

## Commandes

`role-set --help` liste les commandes. Chaque ligne ci-dessous est une commande qui existe aujourd’hui. Les indicateurs sont disponibles dans la documentation de chaque commande `--help`.

| Commande | Ce qu’elle fait |
|---|---|
| `plan` | Définit chaque emplacement pour un lot à partir d’une valeur de départ. Aucune appel de modèle. La même valeur de départ génère le même plan. |
| `cost` | Estime le durée d’exécution et le coût d’un plan avant le lancement (`two-card` ou `frontier`). Il s’agit d’une estimation. |
| `pack sync` / `pack verify` | Copie le paquet de vérification à partir d’un clone local d’Aspire-SI à une version spécifique, et vérifie la copie par rapport à son manifeste. |
| `harvest` | Copie les extraits vérifiés en termes de licence à partir des dépôts référencés dans un pool local. Le répertoire du clone n’est pas enregistré. |
| `write` | L’exécuteur. `--mock` l’exécute sans modèle ni coût. Une exécution réelle nécessite une boucle `--endpoint` et une `--model` à partir de `writers.json`. |
| `readout` | Génère un rapport pour chaque exécution par leçon et par type, indiquant si l’exécution a réussi ou échoué, à partir du répertoire qui contient `batch-report.json`. |
| `critic-control` | Évalue le critique sur les éléments examinés. Nécessite une autorisation et un fichier de signal de présence. `--trait-leg` et `--answer-leg` exécutent ces étapes. |
| `ingest` / `store` | Enregistre les reçus dans le stockage, avec le pare-feu à l’entrée. `store status` affiche les décomptes. `store void` ajoute une annulation. |
| `pilot-auto` | Le lanceur payant : attend la boucle, écrit et peut relancer après un arrêt dû à une interruption. |
| `puzzle-check` | Exécute le solveur de type B sur les fichiers de données et leurs fichiers secondaires `.models.jsonl`, et refuse un squelette de faits répété. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

Le générateur n’a pas encore de commande `role-set`. Ses trois scripts créent les tables validées et l’enregistrement :

| Script | Ce qu’elle fait |
|---|---|
| `scripts/t10_usable_cells.py` | Teste chaque cellule et écrit `t10_cells.json`. `--check` exécute le contrôle de capacité sur la table validée et ne teste pas. |
| `scripts/t10_cell_capacity.py` | Compte les squelettes distincts par cellule sur les données de réglage 1, 21 et 41, par strate, dans `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Reconstruit chaque puzzle enregistré dans `t10_record.json`. `--check` vérifie l’enregistrement validé et affiche les chiffres de la graine verrouillée. |

## Contrôles

Cinq types :

- **A.** Un passage d’enseignement. Pas de verdict. Un validateur s’exécute, et la vérification des caractéristiques est indicative.
- **B.** Un puzzle d’affectation finie. Le solveur vérifie le modèle. La réponse du critique est indicative.
- **C.** Un vérificateur avec un défaut intentionnel, ou un vérificateur qui avait raison. Les vérifications du verdict et des preuves sont refusées. La vérification du défaut est enregistrée et ne refuse pas.
- **D.** Deux tours. Les vérifications du verdict et des preuves sont refusées. La vérification des caractéristiques est indicative.
- **E.** Un quiz écrit, ou un quiz de verdict. Un E écrit n’a pas de vérification de verdict. Un E de type verdict vérifie le verdict et les preuves.

Les vérifications du verdict et des preuves sont refusées. Les vérifications des caractéristiques, des réponses et des défauts sont enregistrées et ne refusent pas, jusqu’à ce que leurs contrôles soient réussis. Le test des caractéristiques a donné un résultat global de 40,3 % par rapport à une barre de 85 %, et la pire caractéristique a donné un résultat de 12,5 % par rapport à 70 %. Le test des réponses a donné un résultat de 62,7 % par rapport à une barre de 90 %. Le contrôle des défauts du 2026-10-11 a donné 9 résultats sur 24.

Le solveur B couvre les puzzles d’affectation finie dans son ensemble de formes de règles : au maximum 6 entités, 3 attributs et 200 000 affectations. Il ne vérifie pas la logique par rapport au modèle. Un élément B basé sur un verdict sur du matériel en texte libre est en dehors du solveur, et `puzzle-check` le considère comme tel, jamais comme réussi.

Le pare-feu du stockage refuse le matériel mis de côté, scellé et utilisé pour les entretiens lors de l’ingestion, et le matériel DEV pendant une étape. Il n’y a pas de chemin de lecture pour ce matériel. La direction est conçue et n’a jamais été exécutée.

## Ce qu’une exécution conserve

Chaque reçu accepté contient le hachage du plan et le hachage du paquet. Lorsque l’exécution a utilisé des exemples, elle contient également le hachage de la validation de l’exemple et le hachage du fichier d’exclusion. Lorsqu’elle a utilisé la règle bloquée, elle contient le hachage de cette règle. Le reçu d’un puzzle généré contient sa graine, ses essais, ses lectures, ses mesures de difficulté et les hachages du squelette, du paquet et du solveur. Chaque appel d’écriture est conservé dans `drafts.jsonl`, qu’il soit accepté ou refusé.

## Modèles

L’épingle de l’écrivain est Qwen3-235B-A22B-Instruct-2507, la version AWQ, Apache-2.0, utilisée comme `qwen3-235b-instruct-2507` à la révision `96548f657d772187f16e9f04bc50aec78a9f9cdd`. L’épingle du critique est `gemma4:31b`, hachage `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, Q4_K_M, Apache-2.0, à `http://127.0.0.1:11434`. `writers.json` répertorie également un écrivain uniquement pour les répétitions et un critique de comparaison. Un nom qui ne figure pas dans le fichier est refusé, tout comme un nom contenant `cloud`.

Le GPU local est utilisé pour le critique et un écrivain de répétition, uniquement dans le cadre d’une autorisation : `--grant` et un fichier de signal de présence modifié il y a moins de 60 secondes. Le fichier de signal de présence n’est pas ouvert. Un critique chargé par l’exécution est déchargé à la fin, y compris en cas d’erreur. `--mock` ne charge rien.

## Pilotes

Trois pilotes payants sont enregistrés dans [docs/runs/](docs/runs/), tous datés du 2026-10-11 : 3,52 $, 3,61 $ et 2,80 $. Ce sont les seuls chiffres de coût mentionnés dans ce fichier README. Aucune ligne de ces données ne figure dans un ensemble d’entraînement. Dans le pilote 2, la vérification des 49 candidats par R&D a validé 15 tels quels, 15 autres après une correction mécanique et a rejeté 19. Le pilote 3 n’a rien écrit.

## Modèle de menace

role-set est un générateur local. Ce n’est pas un serveur. Il n’écoute pas et ne collecte pas de données de télémétrie.

Une exécution touche le plan, le paquet de rôle et le répertoire de sortie que vous nommez. `ingest` et `store` touchent également le fichier de stockage que vous nommez. `write`, `pilot-auto` et `critic-control` touchent un point de terminaison de boucle lorsqu’ils ne sont pas `--mock`. `--mock` n’envoie rien. Le générateur et ses scripts lisent et écrivent des fichiers dans le dépôt et n’appellent aucun modèle.

Une exécution ne touche pas un service de modèle cloud, un pod, un compte de location ou une clé d’API de location. Elle ne lit pas les lignes scellées, les ensembles mis de côté ou les enregistrements d’entretiens. `ingest` lit les douze questions publiques de l’entretien à partir du clone d’Aspire-SI que vous nommez, à la version que vous nommez, uniquement pour refuser une ligne qui les répète. Elles ne sont pas stockées. `store status` renvoie des décomptes, pas des lignes. Signalez une vulnérabilité via le contact dans [SECURITY.md](SECURITY.md).

## Tests

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La couverture des lignes a un seuil de 90, dans `pyproject.toml` et `codecov.yml`, et l’exécution échoue en dessous de ce seuil. Elle mesure environ 97 %. Le badge de couverture ci-dessus affiche la valeur actuelle. CI exécute également la vérification Atlas, `role-set plan` et `puzzle-check` sur un élément de test, sans modèle. `scripts/verify` exécute les vérifications locales dans une seule commande.

Le manuel est publié avec la [page d’accueil](https://mcp-tool-shop-org.github.io/role-set/). Ce fichier README est autonome.

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>.
