# CryptoML — Documentation produit et technique

> **Statut : spécification cible, pas application déjà développée.** Version 1.0 du dossier — 12 octobre 2026. Langue : français. Éléments d'API et de fichiers proposés à implémenter.

CryptoML est une application de recherche en apprentissage automatique pour **anticiper la direction de marchés crypto perpétuels USDⓈ-M en M5** à partir de données historiques multi-actifs, avec un **modèle global LightGBM**, des évaluations rigoureuses et un **replay visuel « prédit vs réel »**.

**Objectif de la première phase : qualité prédictive, pas profits/trading.** Les fonctionnalités de transactions, d'effet de levier et de PnL restent dans le wireframe mais **ne sont pas au périmètre du MVP prédictif**. Aucun passage d'ordre réel n'est prévu.

## Décisions confirmées

- **Interface** : Next.js, TypeScript, shadcn/ui ; TradingView Lightweight Charts pour chandeliers/courbes temporelles ; ECharts pour statistiques.
- **Backend** : Python et FastAPI. Entraînement asynchrone dans un processus worker indépendant.
- **Modèle** : LightGBM, classification directionnelle **HAUSSE / NEUTRE / BAISSE**.
- **Marché** : contrats perpétuels crypto USDT-M (source initiale Binance Futures), sans exécution.
- **Données** : M5, OHLCV Futures et features relatives ; contexte BTC/ETH lorsque synchronisé et disponible.
- **Généralisation** : un modèle global multi-crypto, initialement sans identifiant de symbole comme feature ; tests sur paires jamais vues.
- **Horizons** : 3, 6 et 12 bougies M5 = 15, 30 et 60 minutes ; un modèle par horizon.
- **Évaluation** : séparation chronologique, purge des labels, walk-forward, holdout symboles, calibration et baselines.
- **Produit** : 11 pages, dont « Qualité prédictive » avec replay prédit vs réel sans trades.

## Cartographie des documents

| Document | Contenu |
|---|---|
| [00 — Produit et périmètre](docs/00_PRODUIT_ET_PERIMETRE.md) | Vision, utilisateurs, objectifs, exclusions |
| [01 — Architecture](docs/01_ARCHITECTURE.md) | Modules, flux, structure du dépôt |
| [02 — Données](docs/02_DONNEES_INGESTION.md) | Sources, schémas, qualité, alignement temporel |
| [03 — Features et labels](docs/03_FEATURES_ET_LABELS.md) | Prédictions M5, règles anti-fuite |
| [04 — Entraînement](docs/04_ENTRAINEMENT_LIGHTGBM.md) | Pipeline ML, paramètres, versionnement |
| [05 — Validation](docs/05_VALIDATION_ET_METRIQUES.md) | Walk-forward et cryptos inédites |
| [06 — API](docs/06_CONTRATS_API.md) | Endpoints et format des réponses |
| [07 — Interface](docs/07_PAGES_ET_COMPOSANTS.md) | Spécification des 11 pages |
| [08 — Replay](docs/08_REPLAY_PREDIT_VS_REEL.md) | Chart, simulation point-in-time, interactions |
| [09 — Persistance](docs/09_STOCKAGE_ET_MODELES.md) | PostgreSQL / Parquet / MLflow |
| [10 — Développement](docs/10_ENVIRONNEMENT_DEV.md) | Installation prévue, configuration, lancement |
| [11 — Tests](docs/11_TESTS_ET_CRITERES_ACCEPTATION.md) | Contrôles unitaires, temporels et UX |
| [12 — Roadmap](docs/12_ROADMAP.md) | Phases et livrables |
| [13 — Décisions](docs/13_DECISIONS_OUVERTES.md) | ADR et choix à confirmer |
| [14 — Glossaire et références](docs/14_GLOSSAIRE_ET_SOURCES.md) | Définitions et documentation officielle |

Également inclus : [maquette V4 validée](references/wireframe_lightgbm_crypto_v4_replay.html), [contrat OpenAPI proposé](contracts/openapi.yaml), [exemples de réponses](examples/), [checklist](TASKS_CRYPTOML.md), [instructions à un assistant de code](AGENTS_CRYPTOML.md), [variables d'environnement exemple](.env.cryptoml.example).

## Installation dans un projet VS Code existant

1. Décompresser le ZIP et copier **`docs/`, `contracts/`, `examples/`, `README_CRYPTOML.md`, `TASKS_CRYPTOML.md`, `AGENTS_CRYPTOML.md`, `.env.cryptoml.example`** à la racine du dépôt (sans écraser d'éventuels fichiers existants).
2. Ouvrir `README_CRYPTOML.md` (aperçu Markdown dans VS Code : `Ctrl+Shift+V`).
3. Commencer par les tâches **P0** dans `TASKS_CRYPTOML.md`.
4. Si tu utilises un agent de développement, lui fournir `AGENTS_CRYPTOML.md`, puis le document relatif à la tâche.

> Le dossier contient **de la documentation et un projet de contrat**, pas un backend opérationnel. Les commandes d'exécution décrites dans la documentation ne seront utilisables qu'une fois les modules implémentés.

## Premier jalon recommandé

Importer **BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT et ADAUSDT en M5 Futures**, construire les features passées uniquement, entraîner **H=6 / 30 minutes**, générer un rapport de classification hors échantillon et exposer un replay temporel fidèle. Ensuite étendre à H=3/H=12 et aux 20–30 paires.

## Règles non négociables

1. Pas de donnée future dans les features, les transformations, la validation ni le replay.
2. Les valeurs fictives du wireframe ne sont pas des performances mesurées.
3. Un modèle n'est pas « bon » seulement parce qu'il bat le hasard sur le train ; il faut une amélioration hors échantillon par rapport aux baselines.
4. Les données sources, les paramètres, les versions et les prédictions doivent rester traçables.
5. Aucun ordre réel, clé API privée d'exchange ou levier réel pendant la phase prédictive.
