# P4 V3 — Recensement EMA9/EMA26 sur 21 mois

**Descriptif seulement : sans labels, rendements, entraînement ou nouveaux tests OOS.** Version distincte des recensements V1/V2 sur sept mois.

Entrées : 84 partitions P0 SHA-256 vérifiées pour BTC, ETH, SOL, XRP de **2024-10 à 2026-06** inclus. Exclusion stricte de ADA et des mois 2026-07, 08 et 09. Le détecteur V1, les catégories ATR de régime V2, les conditions de signal, ainsi que les périodes protégées restent inchangés.

## Windows

```powershell
cd E:\cryptoML-trend-pullback-v3
git status --short
git fetch origin
# Si le worktree est propre, récupérer la version la plus récente :
git switch --detach origin/research/trend-pullback-long-history-v3
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_long_census_v3.py tests/test_pullback_long_history_v3.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_long_census_v3 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v3_21month_unlabeled_census_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v3_21month_unlabeled_census_v1.csv"
```

Les fichiers de sortie doivent être **nouveaux**. Aucun Parquet d'origine n'est modifié.

## Résultat et précautions

Le rapport affiche, pour chaque crypto :
- 21 mois et la somme, puis **14 mois anciens** (octobre 2024–novembre 2025) et **7 mois de recouvrement** (décembre 2025–juin 2026) ;
- nombre de bougies finalisées, événements bruts, événements après cooldown, signaux classifiés/non classifiés ;
- tableau croisé des 3 forces × 4 profondeurs EMA9/EMA26, classes fixes V2, vérifié par crypto, mois, sous-période et total.

Le comptage des sept mois de recouvrement peut différer très légèrement de 6 814 : V2 recommençait EMA26/EMA9 à l'ouverture de décembre 2025, V3 bénéficie de l'état antérieur de ses indicateurs et de son cooldown. Cela n'autorise pas à modifier les règles ou à sélectionner les classes supposées favorables.

Si les tests ou le chargement échouent, **ne pas retirer silencieusement un mois, une crypto ou une vérification d'intégrité** ; auditer le problème. L'étude des labels et une extension du protocole Ridge V2 demanderaient un nouveau gel explicite avec une évaluation temporelle vérifiable.
