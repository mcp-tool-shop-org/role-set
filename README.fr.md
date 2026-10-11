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

**Prototype, développé en mode ouvert.** Conseils et expérimentation : les interfaces et les structures de données sont modifiées sans préavis.

Le jeu de rôles définit les éléments de définition des rôles, écrit les parties que l’auteur doit écrire et enregistre les étapes que chaque élément a franchies. Il s’agit d’un générateur, pas d’un juge. Les caractéristiques d’un bon vérificateur se trouvent dans le paquet de rôles, que les humains rédigent. Rien de ce que ce programme écrit ne fait partie d’un ensemble d’apprentissage. Chaque élément passe par la vérification des éléments clés de R&D et une deuxième lecture avant d’être traité.

L’auteur et le critique ne sont accessibles qu’à l’adresse `http://127.0.0.1`, via un port explicite. Tout autre accès est refusé avant l’envoi d’un octet. Le jeu de rôles ne loue jamais de machine et n’accède jamais à un pod. Lorsqu’une exécution utilise un pod, c’est offrig qui le loue, et le jeu de rôles ne voit que le tunnel de boucle.

La conception est décrite dans le document [docs/role-set-design.md](docs/role-set-design.md), et elle est conforme aux exigences du programme. Les notes relatives à l’exécution se trouvent dans le dossier [docs/runs/](docs/runs/).

## Installation

Python 3.12, géré par uv. Une seule dépendance d’exécution : [aspire-si-gates](https://github.com/mcp-tool-shop-org/aspire-si), fixée à une version spécifique d’aspire-si. Les portes sont importées, et non copiées.

```text
uv sync --frozen
```

## Commandes

`role-set --help` liste les commandes. Chaque ligne ci-dessous est une commande qui existe aujourd’hui.

`plan` fixe tous les emplacements pour un lot à partir d’une base. Elle n’appelle pas de modèle. La même base reproduit le plan.

```text
uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json
```

`cost` estime le nombre d’heures et le coût en dollars pour ce plan avant le lancement de quoi que ce soit. Le chiffre est une estimation. Les profils sont `two-card` et `frontier`.

```text
uv run role-set cost --plan plan.json --profile two-card
```

`pack sync` copie le paquet de rôles à partir d’un clone local d’aspire-si à une version spécifique. `pack verify` vérifie cette copie par rapport à son manifeste. Le chemin du clone n’est pas enregistré. Un nouveau clone de ce dépôt n’est pas un paquet synchronisé : `roles/verifier` contient la liste d’exclusion et la liste des sources, et la synchronisation remplit le reste.

```text
uv run role-set pack sync --repo <aspire-si clone> --commit <sha>
uv run role-set pack verify --pack roles/verifier
```

`harvest` copie les extraits dont la licence a été vérifiée dans un pool local. Le répertoire du clone n’est pas enregistré.

```text
uv run role-set harvest --allow-list allow.json --clones <clone directory>
```

`write` est l’exécuteur. `--mock` exécute l’ensemble du pipeline sans modèle et sans coût, et ne demande pas de subvention. La sortie simulée est marquée, ce n’est pas une donnée, et elle est écrite sous `<out>/mock/`. Une exécution qui appelle un modèle nécessite `--endpoint` et `--model`, et le point de terminaison doit être en boucle. Le chargement du critique, ou d’un rédacteur local pour une répétition, nécessite également `--grant` et `--heartbeat`. `--no-critic` est une exécution à blanc : les éléments passent sans la deuxième lecture, et le rédacteur du pod n’utilise alors pas le GPU local.

```text
uv run role-set write --plan plan.json --pack roles/verifier --out out --mock
```

`readout` génère un rapport pour chaque exécution par leçon et par type, y compris une exécution qui n’a accepté aucun élément. Il lit le répertoire qui contient `batch-report.json`. Un `write` terminé laisse ce fichier. `readout` est la commande qui le transforme en rapport.

```text
uv run role-set readout --run out/mock
```

`critic-control` évalue le critique sur les éléments évalués. Elle nécessite une subvention et un fichier de signal de présence. `--trait-leg` et `--answer-leg` exécutent ces deux évaluations.

```text
uv run role-set critic-control --items reviewed.jsonl --grant <grant> --heartbeat <heartbeat file>
```

`ingest` enregistre les reçus dans le stockage. Le pare-feu fonctionne à l’entrée. `store status` signale le fichier. `store void` ajoute une annulation pour un tour. Les lecteurs ignorent un tour annulé. Elle ne supprime pas les lignes.

```text
uv run role-set ingest --receipts <receipts directory> --store store.sqlite --repo <aspire-si clone> --commit <sha>
uv run role-set store status --store store.sqlite
```

`pilot-auto` est le lanceur payant : une seule publication, puis il attend le canal de boucle, écrit et peut relancer après un arrêt dû à une perte de connexion. Les tests ne l’appellent jamais sur un point de terminaison actif. CI utilise `write --mock`.

```text
uv run role-set pilot-auto <port> <grant> <HH:MM UTC> <work directory>
```

`puzzle-check` exécute le solveur de type B sur un fichier de données et le module complémentaire `.models.jsonl` à côté. Fournissez-lui tous les fichiers de données du répertoire en même temps : un puzzle dont le squelette factuel répète un puzzle précédent, avec les noms, l’ordre des lignes et la direction mis de côté, est refusé, en indiquant le premier.

```text
uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl
```

## Portes

Cinq types :

- **A.** Un passage d’enseignement. Pas de verdict. Un validateur s’exécute, et la vérification des traits est indicative.
- **B.** Un puzzle d’affectation finie. Le solveur vérifie le modèle. La réponse du critique est indicative.
- **C.** Un vérificateur avec un défaut intentionnel, ou un vérificateur qui avait raison. Les vérifications du verdict et des preuves refusent. La vérification du défaut est enregistrée et ne refuse pas.
- **D.** Deux tours. Les vérifications du verdict et des preuves refusent. La vérification des traits est indicative.
- **E.** Un questionnaire écrit, ou un questionnaire de verdict. Un E écrit n’a pas de vérification de verdict. Un E de type verdict vérifie le verdict et les preuves.

Les vérifications du verdict et des preuves refusent. Les vérifications des traits, des réponses et des défauts sont enregistrées et ne refusent pas, jusqu’à ce que leurs contrôles soient réussis. La partie des traits a obtenu un score global de 40,3 % par rapport à une barre de 85 %, et le pire trait a obtenu 12,5 % par rapport à 70 %. La partie des réponses a obtenu un score de 62,7 % par rapport à une barre de 90 %. Le contrôle des défauts du 11 octobre 2026 a donné 9 sur 24. L’utilisation des traits ou des réponses pour bloquer les éléments sur la base de ces chiffres refuserait les éléments déjà connus pour être bons.

Le solveur B couvre les puzzles d’affectation finie écrits dans son ensemble de règles fermé : au maximum 6 entités, 3 attributs et 200 000 affectations. Il ne vérifie pas le raisonnement par rapport au modèle. Un élément B basé sur un verdict et utilisant du matériel en texte libre est en dehors du champ d’application du solveur. `puzzle-check` considère cet élément comme étant en dehors du champ d’application, et non comme étant réussi.

Le pare-feu du stockage refuse le matériel mis de côté, scellé et utilisé pour les entretiens lors de l’ingestion, et il refuse le matériel DEV pendant une étape. Il n’existe aucun chemin de lecture pour le matériel mis de côté, scellé ou utilisé pour les entretiens. Le stockage, l’ingestion et le pare-feu sont construits. La direction est conçue et n’a jamais été exécutée.

## Ce qu’une exécution conserve

Chaque reçu accepté contient le hachage du plan et le hachage du paquet. Lorsque l’exécution a utilisé des exemples, le reçu contient également le hachage de la version de l’exemple et le hachage du fichier d’exclusion. Lorsqu’elle a utilisé la règle figée, le reçu contient le hachage de cette règle. Une exécution qui n’accepte aucun élément n’a pas de reçus. Chaque appel au rédacteur est conservé dans `drafts.jsonl`, qu’il soit accepté ou refusé. Un `batch-report.json` terminé laisse ce fichier, et `readout` signale ce répertoire.

## Modèles

Le modèle du rédacteur est Qwen3-235B-A22B-Instruct-2507, la version AWQ, Apache-2.0. `writers.json` le sert en tant que `qwen3-235b-instruct-2507` à la version `96548f657d772187f16e9f04bc50aec78a9f9cdd`.

Le critique est `gemma4:31b`, le digest est `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`, la quantification est Q4_K_M, Apache-2.0. Son point d’accès dans `writers.json` est `http://127.0.0.1:11434`.

`writers.json` liste également un rédacteur utilisé uniquement pour les tests préliminaires, `qwen3:8b`, et un critique comparatif, `granite4.1:30b` (IBM Granite 4.1, Apache-2.0). `write` refuse le critique comparatif en tant que critique pour l’exécution. Un nom qui ne figure pas dans le fichier est refusé. Un nom contenant `cloud` est refusé.

Le GPU local est utilisé pour le critique et pour un rédacteur utilisé pour les tests préliminaires, et uniquement dans le cadre d’une autorisation. Ces chemins nécessitent `--grant` et un fichier de signal de présence dont le temps de modification est inférieur à 60 secondes. `critic-control` exige les deux pour chaque exécution. Le fichier de signal de présence n’est pas ouvert. Lorsqu’une exécution charge le critique, celui-ci est déchargé à la fin, y compris lorsque l’exécution s’arrête avec une erreur. `--mock` ne charge rien.

## Tests pilotes

Trois tests pilotes payés sont enregistrés dans [docs/runs/](docs/runs/). Ce sont les seuls chiffres de coût mentionnés dans ce fichier README. Aucun résultat généré par ces tests ne figure dans un ensemble d’entraînement.

- **Test pilote 1, 2026-10-11, 3,52 $.** Aucune des 104 lignes acceptées n’est une donnée. Le service R&D a considéré les 43 lignes C et D comme mal assemblées avant toute vérification du contenu. La note est disponible à l’adresse [docs/runs/2026-10-11-235b-pilot.md](docs/runs/2026-10-11-235b-pilot.md).
- **Test pilote 2, 2026-10-11, 3,61 $.** Vérification des 49 candidats par le service R&D : 15 ont été validés tels quels, 15 autres nécessitaient une correction mécanique et 19 ont échoué. La note est disponible à l’adresse [docs/runs/2026-10-11-pilot-2.md](docs/runs/2026-10-11-pilot-2.md).
- **Test pilote 3, 2026-10-11, 2,80 $.** Il n’a rien écrit. Le moteur du rédacteur s’est bloqué après le démarrage. La note est disponible à l’adresse [docs/runs/2026-10-11-pilot-3.md](docs/runs/2026-10-11-pilot-3.md).

## Modèle de menace

role-set est un générateur local. Ce n’est pas un serveur. Il n’écoute pas et ne collecte pas de données de télémétrie.

Une exécution affecte le plan, le pack de rôles et le répertoire de sortie que vous nommez. `ingest` et `store` affectent également le fichier de stockage que vous nommez. `write`, `pilot-auto` et `critic-control` affectent un point d’accès en boucle lorsqu’ils ne sont pas `--mock`. `--mock` n’envoie rien.

Une exécution n’affecte pas un service de modèle cloud, un pod, un compte de location ou une clé d’API de location. Elle ne lit pas les lignes scellées, les ensembles mis de côté ou les données d’entretien. `ingest` lit les douze questions publiques de l’entretien à partir du clone aspire-si que vous nommez, à la version que vous nommez, et uniquement afin de pouvoir refuser une ligne qui les répète. Ces questions ne sont pas stockées. `store status` renvoie des décomptes, pas des lignes. Signalez une vulnérabilité via le contact indiqué dans [SECURITY.md](SECURITY.md).

## Tests

```text
uv sync --frozen
uv run pytest
```

Dans cet arbre, il y a 241 tests et une couverture de 97,64 %. Le seuil minimum est de 90, dans `pyproject.toml` et dans CI, et l’exécution échoue en dessous. CI exécute également `role-set plan` et `puzzle-check` sur un ensemble de données de test, sans modèle. `scripts/verify` exécute les mêmes vérifications locales dans une seule commande.

Le manuel est publié avec la [page d’accueil](https://mcp-tool-shop-org.github.io/role-set/). Ce fichier README est autonome.

Créé par <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
