# CryptoML V9 — Mode d'emploi du scanner papier cassure / retest

Nouveau protocole pré-enregistré : [89_P4_V9_BREAKOUT_RETEST_PROSPECTIVE_PROTOCOL.md](89_P4_V9_BREAKOUT_RETEST_PROSPECTIVE_PROTOCOL.md). V8 avait conclu **aucun avantage exploitable** de l'ancien pullback. V9 est une **nouvelle hypothèse**, pas une stratégie rentable prouvée.

## Installer sans toucher aux anciennes branches

\`\`\`powershell
cd E:\cryptoML-long-short-economics-v8
git fetch origin
git worktree add E:\cryptoML-structure-v9 origin/research/structure-breakout-retest-prospective-v9
cd E:\cryptoML-structure-v9\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_structure_breakout_retest_v9.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Tous les nouveaux tests doivent être verts avant un usage expérimental.** Ne lancer ni trading réel ni modèle automatique.

## Entrée CSV réelle, provenant d'une ingestion indépendante

Fournir **un symbole à la fois** : BTCUSDT, ETHUSDT, SOLUSDT ou XRPUSDT. Le fichier M5 doit contenir les colonnes \`symbol,bar_open_at,bar_close_at,open,high,low,close,is_final\`. Horodatages UTC et bougies finales. Minimum 400 bougies d'échauffement ; suffisamment de données M15/H1 continuées (le scanner fait l'agrégation). Le scanner ne télécharge ni ne fabrique de données et **ne transforme pas** les archives historiques V1–V8 en validation indépendante.

Exemple **uniquement quand un CSV source post-verrouillage réel existe** :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.structure_breakout_retest_v9 \`
  --input-csv "E:\cryptoML-walkforward-results\v9_btc_m5_real_closes.csv" \`
  --output-json "E:\cryptoML-walkforward-results\v9_btc_offline_paper_signals_v1.json"
\`\`\`

Le scanner n'émet aucun résultat dont la décision est antérieure au **22 octobre 2026 à 00:00 UTC**. Il émet obligatoirement \`OFFLINE_REPLAY_NOT_LIVE_ATTESTED\`. **Même si toutes les données sont nouvelles**, une analyse hors ligne a posteriori **n'atteste pas** que la décision a été enregistrée au moment du signal. Il faut un journal append-only, horodaté à chaque nouvelle bougie réellement close et vérifié hors du contrôleur d'exécution pour qualifier la suite de réellement prospective.

Aucun CSV P0 ancien n'est ici présenté comme observation future, et **aucune donnée juillet–septembre 2026 n'est utilisée dans le programme** sans sélection explicite de l'opérateur (elles ne sauraient être de l'OOS).

## Résultat et limite

Le JSON contient uniquement : sens, horodatage de cassure, horodatage de confirmation du retest, niveau M15 gelé, ATR14 connu à la cassure et classe H1. **Pas de stop loss ou take profit**, ni portefeuille, ni frais, ni PnL. L'absence de signal ne justifie pas une modification des critères.

Le protocole exige une deuxième étape *avant toute collecte d'issues économiques* : définir un portefeuille papier, les fills plausibles, spread, commissions et funding des perpétuels, latence de décision, taille et limites d'exposition, et les tests de continuité/provenance. Ces règles doivent être datées et verrouillées **avant** de regarder de quelconques rendements du futur échantillon.

**Statut : préparation technique uniquement, tests locaux et collecte authentifiée encore à réaliser.**
