# P2 — Mode d'emploi : H1 à H6, EMA9/EMA26 vs 30 features

**19 octobre 2026 ; uniquement checkout isolé et checks de développement historiquement déjà inspectés.**  
**Protocole figé** : \`docs/49_P2_H1_H6_EMA926_HORIZON_GRID_PROTOCOL.md\`.  
**Module** : \`backend/mlpipeline/ema_horizon_grid.py\`.

Le programme construit, pour chacun des 36 folds autorisés, des prédictions **H1, H2, H3, H4, H5, H6**, avec deux LightGBM multiclasses (EMA9/26 minimal **2 features** vs référence **30 features**) et une règle simple de direction de tendance EMA9 > EMA26. **12 entraînements CPU par fold, 432 au total.**

Les prédictions de H2..H6 représentent la variation cumulée entre le close t et le close t+H ; **H1** mesure aussi explicitement le signe de la **prochaine bougie**, en excluant FLAT du dénominateur et en publiant le nombre FLAT.

Pour toutes les variantes, \`all_check\` force un signe UP/DOWN sur la même ligne ; NEUTRAL canonique est une erreur. \`shared_baseline_selected\` choisit uniquement des lignes sélectionnées par le modèle 30 à partir des scores tuning sans labels (cible 40 %) et applique ce même masque aux trois règles. \`ema2_autonomous\` et \`ema_rule_autonomous\` définissent leurs propres masques de tuning et ont des dénominateurs distincts : comparaison **non appariée**. Ne pas mélanger ces chiffres ni appeler les règles des stratégies rentables.

Le modèle 30 doit reproduire pour H2/H6 les comptes directionnels du rapport ancien **fold par fold** ; tout écart arrête l'exécution. Le même moteur fixe les partitions train 120j, early 7j, tuning 3j, check 4j, **purge H12 60 min** et tronque les chandelles avant construction de label à la limite de check de chaque fold. Ne scanne pas les tests OOS de folds, ni le dernier holdout 90j, ni ADA. Pas de modification P3 ou de modèle figé. Aucun ordre de trading.

## A. Mise à jour et tests

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_ema_horizon_grid.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Si un test échoue : ne pas lancer de smoke et transmettre la trace.** Ne jamais activer le venv de production \`E:\cryptoML\backend\.venv\`.

## B. Smoke (1 fold, 12 LightGBM, fichiers nouveaux)

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.ema_horizon_grid \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" \`
  --output-json "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_smoke_v1.json" \`
  --output-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_smoke_predictions_v1.parquet" \`
  --months 24 --max-folds 1 --verify-hashes
\`\`\`

**Attention** : en cas d'interruption entre l'écriture du Parquet et celle du JSON, le Parquet peut exister seul ; utiliser dans ce cas un **nouveau** suffixe \`_v2\` pour les deux sorties, sans les écraser.

Le smoke doit :
- calculer les six horizons et produire les deux fichiers ;
- faire apparaître H1 à H6 et des prédictions individuelles dans le Parquet ;
- ne lever **aucune** exception de drift H2/H6 ;
- ne donner lieu à **aucune** conclusion de performance fondée sur un seul fold.

## C. Full uniquement après tests verts + smoke réussi

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.ema_horizon_grid \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" \`
  --output-json "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --output-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --months 24 --verify-hashes
\`\`\`

## D. Résumé à partager après le full

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = foreach ($h in 1..6) {
    $s=$r.summary."H$h"
    $view=$s.views.shared_baseline_selected
    [PSCustomObject]@{
        H="H$h"; Minutes=$s.minutes_ahead
        Folds=$view.lightgbm_30.folds
        Precision30Pct=[math]::Round(100*$view.lightgbm_30.mean_precision,3)
        PrecisionEMA2Pct=[math]::Round(100*$view.lightgbm_ema2.mean_precision,3)
        PrecisionReglePct=[math]::Round(100*$view.ema_cross_rule.mean_precision,3)
        DeltaEMA2vs30Pts=[math]::Round($s.mean_delta_ema2_minus_30_shared_points,3)
        FoldsEMA2Meilleur=$s.folds_ema2_wins
        Logloss30=[math]::Round($s.mean_logloss_30,4)
        LoglossEMA2=[math]::Round($s.mean_logloss_ema2,4)
    }
}
$rows | Format-Table -AutoSize
$r.summary.H1.next_bar_sign | Format-List
\`\`\`

## E. Inspecter les six prédictions d'une clôture

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -c "import pandas as pd; p=pd.read_parquet(r'E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet'); x=p[p.symbol.eq('BTCUSDT')]; fold=x.fold_id.min(); moment=x.loc[x.fold_id.eq(fold),'bar_close_at'].min(); print(x[(x.fold_id==fold)&(x.bar_close_at==moment)].sort_values('horizon').to_string(index=False))"
\`\`\`

Une ligne du Parquet représente **un seul horizon, un seul actif, une seule clôture, un seul fold**. Les colonnes \`target_3class\` et \`next_bar_return\` ne sont présentes que pour **l'analyse rétrospective** et ne doivent **jamais** entrer dans les prédictions au moment t.

**Pas de promotion, pas de sélection après-coup du meilleur horizon en vue d'un déploiement.** Les checks ayant déjà été consultés, toute stratégie doit être confirmée sur données chronologiquement nouvelles avant d'être prise au sérieux.
