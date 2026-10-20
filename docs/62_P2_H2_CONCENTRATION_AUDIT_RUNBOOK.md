# P2 H2 sélectif — audit sans entraînement : mode d'emploi

Objectif : expliquer la concentration de l'avantage +0,1355 bp/trade à 2 bp fixes, surtout dans le fold 1. Règles et données inchangées : LightGBM 30 features, LONG/SHORT pour prédictions au-delà de ±3 bp ; sinon abstention. Pas de nouvelle cible, seuil, optimisation, apprentissage, trading ni test 72 h.

**Données strictement nécessaires :** le JSON, le Parquet des prédictions de l'étude H2 retardée déjà terminée et le plan P2 original. Le script ne charge même pas P0. Il refuse les décalages de clés, de folds, de dates, de maturité et de résultats par rapport au rapport original. Les P3, ADA, OOS réservés et final90 restent intacts.

## 1 — Télécharger la dernière branche et tester

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h2_selective_concentration_audit.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

## 2 — Audit complet (pas de fits)

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h2_selective_concentration_audit `
  --binary-report "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_full_v1.json" `
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_predictions_v1.parquet" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output-json "E:\cryptoML-walkforward-results\p2_h2_concentration_audit_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p2_h2_concentration_audit_v1.csv"
```

Les deux fichiers de sortie doivent être nouveaux.

## 3 — Lire l'audit

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_h2_concentration_audit_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.all
$r.fold_1
$r.without_fold_1
$r.positive_folds
$r.negative_folds
$r.empty_folds
$r.deduplicated_earliest_fold_diagnostic
$r.extremes.outlier_sensitivity | Format-Table -AutoSize
$r.by_side | Format-List
$r.by_asset | Format-List
```

Les sommes de bp sont des **sommes de markouts**, pas un PnL de portefeuille. Les folds sont historiquement chevauchants ; la vue dédupliquée ne reproduit pas un moteur d'exécution. Les extrêmes et le leave-one-fold-out sont des *stress tests descriptifs*, pas des règles de sélection. Un résultat positif sur les anciens checks ne prouve ni stabilité hors-échantillon ni rentabilité.
