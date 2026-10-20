# P2 — Sélection interne H2 : procédure Windows isolée

Voir `docs/63_P2_INTERNAL_SELECTION_TRAIN_EARLY_TUNING_PROTOCOL.md`.

**Important :** ce run utilise uniquement train/early/tuning des 36 folds déjà fixés. Il **n'importe ni l'ancien rapport H1-H6, ni les prédictions ou vérités des checks réutilisés**, et ne produit aucune mesure sur ces checks. Il ne regarde jamais ADA, les folds OOS, le holdout final90 ou P3. P0 est lu **sans écriture**, quatre actifs seulement, fingerprint vérifié. Le tuning de 3 jours, déjà consulté auparavant pour des politiques de couverture, n'est pas indépendant.

## 1. Tests obligatoires

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h2_internal_selection.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Si échec : ne pas lancer l'expérience, fournir la trace.

## 2. Exécution 36 folds train / early / tuning uniquement

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h2_internal_selection `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output-json "E:\cryptoML-walkforward-results\p2_h2_internal_selection_full_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p2_h2_internal_selection_full_v1.csv" `
  --output-parquet "E:\cryptoML-walkforward-results\p2_h2_internal_selection_tuning_predictions_v1.parquet" `
  --verify-hashes
```

Les sorties doivent être nouvelles, sous `E:\cryptoML-walkforward-results`. Le calcul entraîne exactement 36 LightGBM et 36 modèles Ridge ; aucune réutilisation de labels de check. La procédure peut conclure `NO_ELIGIBLE_CANDIDATE`.

## 3. Lire les résultats

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_h2_internal_selection_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.selection
$r.summary.regression_30.eligibility_rules | Format-List
$r.summary.ridge_ema2.eligibility_rules | Format-List

Import-Csv "E:\cryptoML-walkforward-results\p2_h2_internal_selection_full_v1.csv" |
  Format-Table -AutoSize
```

**Ne pas** prendre une sélection interne pour une validation économique indépendante ; les signaux sont calculés avec une hypothèse d'entrée future non garantie, et les périodes tuning ne sont pas des observations indépendantes.
