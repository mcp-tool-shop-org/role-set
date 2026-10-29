<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
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

**Prototype, développé en mode ouvert.** Conseils et expérimentation : les interfaces et les structures de données sont modifiées sans préavis. Rien de ce qui est écrit dans l’ensemble des rôles n’est considéré comme des données d’entraînement tant qu’une personne ne l’a pas vérifié. Chaque élément est soumis à la vérification clé du service R&D et à une deuxième lecture avant de pouvoir être intégré à un ensemble d’entraînement.

L’ensemble des rôles planifie les éléments de définition des rôles, écrit les parties qu’un rédacteur doit écrire et enregistre les étapes que chaque élément a franchies. Il crée également des énigmes de type B en code et les valide avant qu’un modèle ne les voie. Il s’agit d’un générateur, pas d’un juge. Ce qui définit un bon rôle se trouve dans un ensemble de rôles, que les personnes rédigent.

Le rédacteur et le critique ne sont accessibles qu’à l’adresse `http://127.0.0.1`, via un port explicite. Tout autre accès est refusé avant l’envoi d’un octet. L’ensemble des rôles ne loue jamais de machine et ne touche jamais à un pod. Lorsqu’une exécution utilise un pod, offrig le loue, et l’ensemble des rôles ne voit que le tunnel de boucle.

Les conceptions sont disponibles à l’adresse [docs/role-set-design.md](docs/role-set-design.md) et [docs/generator-design.md](docs/generator-design.md), et sont maintenues en cohérence avec le programme. Les notes d’exécution se trouvent à l’adresse [docs/runs/](docs/runs/).

## Installation

Python 3.13 ou version ultérieure (3.15 est la cible principale ; les tests CI testent les versions 3.13, 3.14 et 3.15). Une seule dépendance d’exécution : [aspire-si-gates](https://pypi.org/project/aspire-si-gates/) 1.7.6, version fixe. Les étapes sont importées, et non copiées.

```text
pip install role-set
```

Pour travailler sur l’ensemble des rôles lui-même, clonez-le et laissez uv installer l’environnement verrouillé :

```text
uv sync --frozen
```

## Les ensembles de rôles

Un ensemble de rôles est le texte que les personnes ont rédigé pour un rôle. L’ensemble des rôles le fournit en tranches et n’y ajoute rien. Son hachage figure sur chaque reçu accepté.

- **Le vérificateur** (`roles/verifier`). `pack sync` le copie à partir d’un clone aspire-si local à une révision : les leçons, les 18 caractéristiques, le tableau des types, les amendements et les règles. `pack verify` vérifie la copie par rapport à son manifeste. Un clone récent ne contient que la liste des sources, la liste d’exclusion et `defects.jsonl`, le catalogue des classes de défauts identifiées lors des révisions. La synchronisation complète le reste.
- **Le défenseur** (`roles/advocate`). Un ensemble qu’une personne a rédigé dans ce dépôt, sans texte de modèle. Le directeur a approuvé l’ensemble le 2026-10-14 (révision `ffb9546` sur la demande d’extraction n° 56), et le lot 1 est en cours de construction ; il ne figure pas sur `main` tant que la demande d’extraction n° 56 n’est pas fusionnée. Le lot 1 comporte trois caractéristiques : `failed-gate-first`, `exact-figures` et `faithful-relay`. Aucun élément du défenseur n’est considéré comme des données d’entraînement tant qu’il n’a pas passé la lecture pilote.

## Le générateur T10

Les énigmes de type B étaient auparavant rédigées à la main, puis vérifiées. Désormais, le code crée l’énigme et la valide, et un modèle se contente de la formuler. Le générateur est le package `role_set.generate`. T10, la leçon sur les mots de taille stricte (« exactement », « au moins », « au plus »), est la partie qui a changé dans la version 0.2.0. Une recette T12 (quelle édition était en vigueur à une date donnée) est également créée. T11 ne l’est pas.

**Validé par construction.** Le créateur échantillonne une solution cachée, ajoute des indices qui sont vrais dans cette solution jusqu’à ce que la réponse soit définie, et supprime tous les indices dont la réponse n’a pas besoin. Le résolveur vérifie le modèle. Ensuite, le créateur exécute les quatre lectures T10 du service R&D : « exactement » lu comme « au moins », « exactement » lu comme « au plus », « au moins » lu comme « exactement » et « au plus » lu comme « exactement ». Une énigme est conservée uniquement si l’une de ces lectures, appliquée seule à l’indice de caractéristique, modifie la réponse. Une énigme qui réussit sur un autre indice de comptage est la classe de défaut `trait_not_tested`, et elle est rejetée.

**La carte des cellules.** Chaque ensemble, question et valeur de départ est associé à une cellule structurelle : le nombre d’entités `n` (de 3 à 5), le mot de comptage `op`, `k` (de 1 à n-1) et `ifs`, le nombre d’indices `if` (de 0 à 2). Les questions d’ensemble utilisent les trois mots de comptage. Les questions d’affirmation n’utilisent que « exactement », car « au moins » et « au plus » ne peuvent pas modifier le verdict d’une affirmation. Les cellules utilisables sont sondées une fois et validées dans `t10_cells.json`.

**L’axe d’affirmation `if_shape`.** Une cellule d’affirmation définit également la forme de ses `if` indices. Il existe sept formes : `none` pour aucun `if` ; `pos` ou `neg` pour un `if`, en fonction de la polarité de sa conséquence ; et `chain`, `shared_consequent`, `shared_antecedent` ou `disjoint` pour deux. Une construction dont l’énigme réduite perd sa forme tente de nouveau à l’intérieur de la même valeur de départ.

**L’allocateur.** Pour une question et un nombre de valeurs, un fichier de 20 valeurs de départ est attribué une fois, avant que quoi que ce soit ne soit construit, par un flux de coût minimal. Les 20 valeurs de départ de chaque ensemble prennent 20 cellules différentes. Une cellule qui ne contient pas d’indice `if` est utilisée au plus deux fois dans un fichier (la limite zéro-si). Les visites suivantes privilégient les cellules qui ont donné plus de squelettes distincts lors des valeurs de départ de réglage. Ces pondérations sont conservées séparément pour les ensembles à deux et trois valeurs.

**Les seuils de variété.** Dans chaque strate (ensemble et affirmation, deux et trois valeurs) : au moins 75 % de squelettes canoniques distincts, aucun squelette n’apparaît plus de trois fois et aucun squelette n’apparaît deux fois dans le même fichier d’un ensemble. Les seuils sont évalués sur des valeurs de départ verrouillées. Sur les valeurs de départ verrouillées 101 à 120, les affirmations à trois valeurs atteignent 118/120 valeurs distinctes, avec un nombre maximal de répétitions de 2.

**L’enregistrement validé.** `t10_record.json` contient 1 280 constructions (valeurs de départ 1 à 20, 61 à 80, 81 à 100 et 101 à 120, chaque ensemble, les deux questions), chacune avec sa cellule, `if_shape`, ses tentatives et son squelette canonique. Il contient un sha256 sur la source du générateur, la table des cellules, la table des capacités et les données de l’ensemble. Un test garantit que cette liste de fichiers est fermée par les importations de l’ensemble des rôles. Le test CI normal vérifie le hachage et les seuils à partir de l’enregistrement en quelques minutes. La reconstruction exhaustive de chaque énigme s’exécute en tant que tâche `sweep`, uniquement sur `workflow_dispatch`, et échoue en cas de différence par rapport à l’enregistrement.

**La porte de capacité et la marge.** `scripts/t10_usable_cells.py --check` signale la disponibilité, la demande et la marge pour chaque strate, et échoue si un fichier ne peut pas être placé. Avec `if_shape`, les ensembles à trois valeurs disposent de 61 cellules d’affirmation utilisables, contre 27 auparavant. Aucun ensemble n’a besoin d’une visite zéro-`if`, et un test place douze ensembles à trois valeurs (les six réels et six clones) sans répétition dans un même fichier.

## Commandes

`role-set --help` liste les commandes. Chaque ligne ci-dessous est une commande qui existe aujourd’hui. Les indicateurs sont disponibles dans la `--help` de chaque commande.

| Commande | Ce qu’elle fait |
|---|---|
| `plan` | Corrige chaque plage pour un lot à partir d’une graine. Aucune invocation de modèle. La même graine génère le même plan. |
| `cost` | Estime le nombre d’heures et le coût pour un plan avant le lancement (`two-card` ou `frontier`). Il s’agit d’une estimation. |
| `pack sync` / `pack verify` | Copie le paquet de vérification à partir d’un clone local d’Aspire-SI à une version spécifique, et vérifie la copie par rapport à son manifeste. |
| `harvest` | Copie les extraits vérifiés en termes de licence à partir de dépôts référencés dans un pool local. Le répertoire du clone n’est pas enregistré. |
| `write` | Le programme principal. `--mock` l’exécute sans modèle ni coût. Une exécution réelle nécessite une boucle `--endpoint` et une `--model` à partir de `writers.json`. `--wave write`, `critique` et `rewrite` divisent un lot afin que l’évaluateur et le critique s’exécutent l’un après l’autre, sur des hôtes différents si nécessaire, avec `wave-drafts.jsonl` entre eux. |
| `readout` | Fournit un rapport pour chaque exécution par leçon et par type, indiquant si l’exécution a réussi ou échoué, à partir du répertoire qui contient `batch-report.json`. |
| `critic-control` | Évalue un critique sur les éléments examinés, sur son serveur Llama. Nécessite les hachages GGUF et du serveur du critique, ainsi que la ligne de machine, plus une autorisation et un fichier de signalement d’activité à partir d’un hôte loué. `--trait-leg` et `--answer-leg` exécutent ces étapes. |
| `ingest` / `store` | Enregistre les reçus dans le stockage, avec le pare-feu à l’entrée. `store status` fournit les décomptes. `store void` ajoute une annulation. |
| `pilot-auto` | Le lanceur payant des pilotes du 2026-10-11. Son script nomme les dépôts Qwen et Gemma des pilotes, qui ne se trouvent pas dans `writers.json`, donc `write` le refuse. |
| `puzzle-check` | Exécute le solveur de type B sur les fichiers de données et leurs fichiers secondaires `.models.jsonl`, et refuse un schéma de faits répété. |

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
uv run role-set readout --run out/mock
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

Le générateur n’a pas encore de commande `role-set`. Ses trois scripts créent les tables validées et l’enregistrement :

| Script | Ce qu’elle fait |
|---|---|
| `scripts/t10_usable_cells.py` | Teste chaque cellule et écrit `t10_cells.json`. `--check` exécute la validation de capacité sur la table validée et ne teste pas. |
| `scripts/t10_cell_capacity.py` | Compte les schémas distincts par cellule sur les graines de réglage 1, 21 et 41, par strate, dans `t10_capacity.json`. |
| `scripts/t10_variety_record.py` | Reconstruit chaque puzzle enregistré dans `t10_record.json`. `--check` vérifie l’enregistrement validé et affiche les chiffres de la graine verrouillée. |

## Validations

Cinq types :

- **A.** Un passage d’enseignement. Pas de verdict. Un validateur s’exécute, et la vérification des caractéristiques est indicative.
- **B.** Un puzzle d’affectation finie. Le solveur vérifie le modèle. La réponse du critique est indicative.
- **C.** Un vérificateur avec un défaut intentionnel, ou un vérificateur qui avait raison. Les vérifications du verdict et des preuves refusent. La vérification du défaut est enregistrée et ne refuse pas.
- **D.** Deux tours. Les vérifications du verdict et des preuves refusent. La vérification des caractéristiques est indicative.
- **E.** Un quiz écrit, ou un quiz de verdict. Un E écrit n’a pas de vérification de verdict. Un E de type verdict vérifie le verdict et les preuves.

Les vérifications du verdict et des preuves refusent. Les vérifications des caractéristiques, des réponses et des défauts sont enregistrées et ne refusent pas, jusqu’à ce que leurs contrôles soient réussis. La validation des caractéristiques a donné un résultat global de 40,3 % par rapport à une barre de 85 %, et la pire caractéristique a donné un résultat de 12,5 % par rapport à 70 %. La validation des réponses a donné un résultat de 62,7 % par rapport à une barre de 90 %. Le contrôle des défauts du 2026-10-11 avait 9/24.

Le solveur B couvre les puzzles d’affectation finie dans son ensemble de formes de règles : au maximum 6 entités, 3 attributs et 200 000 affectations. Il ne vérifie pas le raisonnement par rapport au modèle. Un élément B basé sur un verdict sur du matériel en texte libre est en dehors du solveur, et `puzzle-check` le considère comme tel, jamais comme réussi.

Le pare-feu du stockage refuse le matériel mis de côté, scellé et utilisé pour les entretiens lors de l’ingestion, et le matériel DEV pendant une étape. Il n’existe aucun chemin de lecture pour ce matériel. La direction est conçue et n’a jamais été exécutée.

## Ce qu’une exécution conserve

Chaque reçu accepté contient le hachage du plan et le hachage du paquet. Lorsque l’exécution a utilisé des exemples, elle contient également le commit de l’exemple et le hachage du fichier d’exclusion. Lorsqu’elle a utilisé la règle bloquée, elle contient le hachage de cette règle. Le reçu d’un puzzle généré contient sa graine, ses essais, ses lectures, ses mesures de difficulté et les hachages du schéma, du paquet et du solveur. Chaque appel d’évaluateur est conservé dans `drafts.jsonl`, qu’il soit accepté ou refusé.

## Modèles

`writers.json` est la liste des évaluateurs. Une utilisation est autorisée, sauf si une clause dans la licence ou les conditions du créateur la restreint ; le silence n’est pas une restriction. Chaque entrée enregistre la recherche des conditions : l’URL de la page, la date à laquelle elle a été lue, qui l’a lue, son `registry_id` dans le registre des licences de rnd (`licences/registry.json`), et son verdict de formation (`allowed`, `conditions` ou `barred`). Une entrée sans cet enregistrement, ou dont le verdict est `barred`, est refusée lorsque la liste est chargée. Un nom qui ne figure pas dans la liste est refusé, tout comme un nom contenant `cloud`.

Les évaluateurs sont Nemotron 3 Ultra 550B, DeepSeek-R1-0528, Nemotron 3.5 Super 120B (ces trois sur un pod), et les huit branches du pilote : Nemotron 3.5 Bolt 74B (la propre version 24 Go de NVIDIA, optimisée pour la quantification), Muse-Glimmer-30B, Nemotron 3.5 Lightning 30B, Qwen3.8-27B, Qwen3.6-35B-A3B, Granite-4.2-30B, Gemma-4-31B-it et DeepSeek-R1-Distill-Qwen-32B. Les critiques sont Muse-Glimmer-30B et Nemotron 3.5 Lightning. Un critique de la même famille que l’évaluateur est refusé. Chaque entrée validée contient un `pin_note` : la façon dont son GGUF a été créé, et si elle correspond à un A100 de 40 Go sur le contexte de 32 768 tokens du pilote.

Les modèles sont servis par `llama-server` de llama.cpp, construit avec CUDA 13.4, à Q8_0. Un modèle répertorié est refusé jusqu’à ce qu’il soit validé : sa révision de dépôt, son fichier GGUF et le sha256 du fichier. Le script wave hache le GGUF et le binaire `llama-server` avant le lancement, et l’exécution refuse toute dérive par rapport à la liste. Chaque reçu enregistre la partition (`production` ou `trial:<name>`), la famille, la révision et le hachage GGUF des deux modèles, le commit et le hachage du serveur, et la ligne de machine CUDA.

**Valeurs des cartes.** L’auteur et le critique utilisent chacun les paramètres documentés de leur créateur, qui sont lus à partir de `experiments/local-builders/card-values.json` avec `--cards` (le fichier J2 est lu, donc un modèle fonctionne de la même manière pour les deux). Chaque clé d’échantillonnage et les paramètres du modèle de conversation sont utilisés pour chaque appel, y compris les valeurs désactivées, de sorte que les valeurs par défaut de `llama-server` n’atteignent jamais un modèle. Le serveur fonctionne avec un contexte de 32 768 tokens et une réponse peut utiliser 16 384 tokens, car chaque modèle doté d’un mode de réflexion est capable de raisonner. Le texte système de R1-Distill est intégré à son tour d’utilisateur, comme l’indique sa carte. Un auteur est associé à sa carte par son `card` ID dans `writers.json`, ou par son nom d’utilisateur. Les reçus enregistrent le sha256 du fichier, le candidat et exactement ce qui a été envoyé, pour l’auteur et le critique. Une requête `write` ou `critic-control` réelle sans `--cards` est refusée.

Une étape ne s’exécute que dans le cadre d’une autorisation, à moins que la ligne de machine n’indique une VM Colab ou un pod : `--grant` et un fichier de signal de présence modifié il y a moins de 60 secondes. Une ligne qui indique le système doit le faire, quels que soient les paramètres de sa carte. Le fichier de signal de présence n’est pas ouvert. `--mock` ne charge rien.

## Tests pilotes

Trois tests pilotes payés sont enregistrés dans [docs/runs/](docs/runs/), tous datés du 11 octobre 2026 : 3,52 $, 3,61 $ et 2,80 $. Ce sont les seuls chiffres de coût mentionnés dans ce fichier README. Aucune de ces lignes ne figure dans un ensemble d’entraînement. Dans le test pilote 2, la vérification des 49 candidats par l’équipe R&D a validé 15 candidats tels quels, 15 autres après une correction technique et a rejeté 19 candidats. Le test pilote 3 n’a rien écrit.

## Modèle de menace

role-set est un générateur local. Ce n’est pas un serveur. Il n’écoute pas et ne collecte pas de données de télémétrie.

Une exécution affecte le plan, le pack de rôles et le répertoire de sortie que vous spécifiez. `ingest` et `store` affectent également le fichier de stockage que vous spécifiez. `write`, `pilot-auto` et `critic-control` affectent un point de terminaison de boucle lorsqu’ils ne sont pas `--mock`. `--mock` n’envoie rien. Le générateur et ses scripts lisent et écrivent des fichiers dans le dépôt et n’appellent aucun modèle.

Une exécution n’affecte pas un service de modèle cloud, un pod, un compte de location ou une clé d’API de location. Elle ne lit pas les lignes scellées, les ensembles mis de côté ou les données d’entretien. `ingest` lit les douze questions publiques de l’entretien à partir du clone aspire-si que vous spécifiez, à la version que vous spécifiez, uniquement pour rejeter une ligne qui les répète. Elles ne sont pas stockées. `store status` renvoie des décomptes, pas des lignes. Signalez une vulnérabilité via le contact indiqué dans [SECURITY.md](SECURITY.md).

## Tests

```text
uv sync --frozen
uv run pytest -m "not sweep"
```

La couverture des lignes a un seuil de 90 %, dans `pyproject.toml` et `codecov.yml`, et l’exécution échoue en dessous de ce seuil. Elle mesure environ 97 %. Le badge de couverture ci-dessus affiche la valeur actuelle. CI exécute également la vérification atlas, `role-set plan` et `puzzle-check` sur un élément de test, sans modèle. `scripts/verify` exécute les vérifications locales dans une seule commande.

## Contributeurs

Certaines parties de ce dépôt ont été écrites avec Grok (xAI). Grok, Argon et Claude sont mentionnés dans [CONTRIBUTORS.md](CONTRIBUTORS.md).

Le manuel est publié avec la [page d’accueil](https://mcp-tool-shop-org.github.io/role-set/). Ce fichier README est autonome.

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
