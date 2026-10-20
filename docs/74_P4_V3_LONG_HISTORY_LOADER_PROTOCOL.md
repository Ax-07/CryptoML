# P4 V3 — Extension sûre de l'historique de développement

**Statut : chargeur et tests uniquement. Aucun entraînement V3, aucun nouveau backtest, aucun résultat économique.**

## Motivation

Inventaire **déclaré par l'opérateur** le 20 octobre 2026 : cinq actifs BTC/ETH/SOL/XRP/ADA, 24 partitions mensuelles de 2024-10 à 2026-09 pour chacun, toutes déclarées `quality_status=OK`. Le dépôt GitHub ne contient pas les fichiers locaux ; les checksums doivent être vérifiés à chaque chargement.

Les premières expériences P4 n'utilisaient que 2025-12–2026-06 parce qu'elles réutilisaient `prospective_model_pack.load_clean_historic`, initialement conçu pour l'entraînement P3 limité aux sept derniers mois de développement. **Ce n'était pas une limitation des EMA ou des données disponibles.**

## Changement isolé sur une branche distincte

`research/trend-pullback-long-history-v3`, dérivée de V2 ; **ne pas modifier les modules de V1/V2 déjà consultés et figés.**

`mlpipeline.pullback_long_history_v3.load_long_development_history` lit seulement :
- 4 symboles autorisés : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT ;
- **21 mois complets de partitions P0**, d'octobre 2024 à juin 2026 ;
- 84 manifestes exacts, qualité `OK`, tailles positives et empreintes SHA-256 vérifiées contre les Parquets avant lecture ;
- M5 définitif, colonnes et géométrie OHLC valides, timestamps UTC, vérification de l'unicité des bougies et absence de chevauchement ;
- la dernière bougie de juin se clôturant exactement au 1er juillet est vérifiée mais exclue du développement pour respecter la limite exclusive UTC ;
- les trous de données restent présents et seront **réinitialisés** par le détecteur V1 ; aucune bougie synthétique ou interpolation.

Le chargeur ne lit **aucun Parquet** ADA ni juillet–septembre 2026, même s'ils existent sur disque. Les métadonnées de manifestes peuvent être inventoriées pour localiser les mois, sans lire les fichiers de bougies réservées.

## Prochaine étape technique

1. Tester le nouveau chargeur exclusivement avec données synthétiques.
2. Si les tests réussissent, étudier les 21 mois en **recensement sans labels** (effectifs et fréquence des signaux, sans mesures de rendement).
3. Définir ensuite **un protocole V3 prédictif séparé**, des partitions temporelles contrôlées, une purge de maturité, et une comparaison à exposition identique (Ridge V1, Ridge augmenté, sélection aléatoire, zéro-trade).
4. Tous les anciens résultats P4/H2 consultés restent du **développement exploratoire**. Les 14 mois supplémentaires améliorent la diversité mais ne constituent pas un nouveau holdout indépendant. Une conclusion confirmatoire requiert un processus prospectif gelé à l'avance.

## Windows — tests synthétiques, aucun accès aux données marché

```powershell
cd E:\cryptoML-trend-pullback-v2
git fetch origin
git worktree add E:\cryptoML-trend-pullback-v3 origin/research/trend-pullback-long-history-v3
cd E:\cryptoML-trend-pullback-v3\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_long_history_v3.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Pas de commande de chargement réel fournie à ce stade : le code doit passer les tests de ton environnement Windows avant toute lecture des 84 partitions.
