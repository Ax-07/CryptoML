# P2 — Comparaison économique H2/H3/H6 (hors ligne)

Cette étude relit les 36 checks de développement **déjà inspectés**, sans réentraînement ni test réel de 72 h. Rendements H2/H3/H6 reconstruits depuis P0 **en lecture seule**, après limitation à chaque borne `check_end`. Même symbole, même timestamp, même horizon pour toutes les références. Aucun ADA, OOS réservé, final90 ou P3.

**Important :** les résultats sont des *markouts* indicatifs entre close(t) et close(t+H), et **PAS des PnL exécutables**. La prédiction n'existe qu'après close(t). Aucun fill réel, carnet d'ordres, funding, capital ni frais effectifs. Hypothèses fixes de coût aller-retour de 0/2/5/10 bp. Comparer séparément la vue dense (positions hypothétiques chevauchantes, non additives) et la phase 0 déterministe (une observation toutes les H bougies par actif et par fold). Les autres phases temporelles sont consignées mais ne doivent pas servir à choisir a posteriori celle qui gagne.

## Vérifier les tests

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h2_h3_h6_markout.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

## Lancer une seule étude complète sans fits

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h2_h3_h6_markout `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" `
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" `
  --output-json "E:\cryptoML-walkforward-results\p2_h2_h3_h6_markout_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p2_h2_h3_h6_markout_v1.csv" `
  --verify-hashes
```

Les sorties doivent être absentes au lancement. En cas d'erreur d'alignement, les labels reconstruits sont rejetés.

## Tableau comparatif

```powershell
Import-Csv "E:\cryptoML-walkforward-results\p2_h2_h3_h6_markout_v1.csv" |
  Where-Object { $_.view -eq "fixed_nonoverlap_phase0" -and $_.cost_bps -eq "2" } |
  Select-Object horizon, model, signals, accuracy_nonflat, gross_mean_bps, net_mean_bps |
  Format-Table -AutoSize
```

Reporter aussi la vue dense séparément. Aucun choix de modèle ou d'horizon n'est justifié comme validé par ces checks.
