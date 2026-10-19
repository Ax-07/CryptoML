# P2 — H1 binaire hors ligne : exécution Windows isolée

Protocole fixe : \`docs/54_P2_H1_BINARY_PROTOCOL.md\`. Cette expérience n'utilise **aucun nouveau test réel de 72 h**. Quatre modèles : LightGBM binaire EMA2, EMA2 + close_location, 30 features, régression logistique EMA2. Les anciens modèles H1 à trois classes sont des références inchangées. Comparaison H1 signe brut, exclusion des bougies futures plates de l'entraînement et de la métrique primaire. Résultats **exploratoires**, anciens checks développement inspectés.

## 1. Tests

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h1_binary_diagnostic.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

Arrêter si tests rouges, et communiquer la trace. Aucun code P3 n'est modifié.

## 2. Smoke, un seul fold, quatre fits

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_binary_diagnostic \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --output-json "E:\cryptoML-walkforward-results\p2_h1_binary_smoke_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p2_h1_binary_smoke_v1.csv" \`
  --output-parquet "E:\cryptoML-walkforward-results\p2_h1_binary_smoke_predictions_v1.parquet" \`
  --months 24 --max-folds 1 --verify-hashes
\`\`\`

Un fold ne démontre rien ; il permet uniquement de constater l'absence de dérive et la disponibilité de toutes les métriques.

## 3. Full, 36 folds / 144 fits

Lancer seulement si tous les tests et le smoke passent :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_binary_diagnostic \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --output-json "E:\cryptoML-walkforward-results\p2_h1_binary_full_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p2_h1_binary_full_v1.csv" \`
  --output-parquet "E:\cryptoML-walkforward-results\p2_h1_binary_full_predictions_v1.parquet" \`
  --months 24 --verify-hashes
\`\`\`

Les sorties doivent être **neuves et sous \`E:\cryptoML-walkforward-results\`**. Lecture seule P0 original : BTC, ETH, SOL, XRP exclusivement. Ni ADA, ni tests OOS du fold, ni final90 ni modèles P3.

## 4. Rapport

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_h1_binary_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.folds_analyzed
$r.summary.accuracy | Format-List
$r.summary.paired | Format-List
$r.summary.by_symbol | Format-List
\`\`\`

Le CSV conserve les huit modèles (4 nouveaux, 4 références). Le Parquet sauvegarde les probabilités binaires et les prédictions sur les anciens checks pour analyses **hors ligne** ; ne pas utiliser les vérités historiques enregistrées pour modifier les paramètres puis qualifier la même période d'OOS indépendante. Les nombres de folds favorables ne sont pas des tests statistiques indépendants.

Les métriques secondaires de probabilité (AUC, Brier, logloss, marges fixées) ne sont calculées que sur les retours futurs non nuls, **pas** sur tous les M5 en temps réel, puisqu'un mouvement nul est inconnu au moment de la prédiction.
