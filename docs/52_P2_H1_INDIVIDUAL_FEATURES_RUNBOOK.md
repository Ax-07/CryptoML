# P2 — Mode d'emploi H1 : 30 indicateurs seuls + EMA2 et un indicateur

**Expérience prédéclarée** : \`docs/51_P2_H1_INDIVIDUAL_FEATURES_PRESPECIFIED.md\`. **Branche isolée** : \`feat/cpu-purged-walkforward-folds\`.

Le modèle cible le signe **du close de la prochaine bougie M5 (H1)**. Il reste entraîné sur les **labels H1 canonique 3 classes**, et la métrique principale est évaluée sur le signe brut de la prochaine bougie non plate : ne pas confondre les deux.

Le script réutilise exactement les anciennes prédictions du LightGBM30 et du LightGBM EMA2, contrôle les clés H1 et les retours calculés à partir des prix, et entraîne 60 modèles candidats par fold : 30 indicateurs chacun seul, puis ajoutés séparément à EMA9/26. **36 folds × 60 = 2160 entraînements CPU**, sans nouveau modèle de référence. Tous les splits restent 120j train / 7j early / 3j tuning score-only / 4j check, purge H12 = 60min. Il ne charge pas ADA, et ne score ni OOS de fold, ni holdout 90 jours, et ne touche pas à P3.

## Tests, avant tout run

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h1_individual_feature_study.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

Ne pas lancer le smoke si ces tests échouent.

## Smoke 1 fold / 1 feature (2 LightGBM CPU)

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_individual_feature_study \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --output-json "E:\cryptoML-walkforward-results\p2_h1_individual_smoke_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p2_h1_individual_smoke_v1.csv" \`
  --months 24 --max-folds 1 --max-features 1 --verify-hashes
\`\`\`

La feature smoke est **la première de la liste gelée** (\`ret_1\`), jamais choisie selon la performance. Le succès signifie seulement absence de dérive des clés/labels et fichiers écrits. Un fold n'apporte pas de résultat scientifique.

## Full 36 folds × 30 features × 2 variantes

Lancer **uniquement après tests verts et smoke réussi** :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_individual_feature_study \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --output-json "E:\cryptoML-walkforward-results\p2_h1_individual_full_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p2_h1_individual_full_v1.csv" \`
  --months 24 --verify-hashes
\`\`\`

Les fichiers de sortie doivent ne pas exister ; **ne jamais écraser un ancien audit**. Le JSON donne les résultats de chaque fold, des quatre actifs, et des vues \`all_nonflat\`, \`shared30_nonflat\`, \`ema2_nonflat\`. Le CSV fournit **les 60 comparaisons**, en ordre fixe du schéma et sans sélection post hoc.

## Extrait de lecture sur console

\`\`\`powershell
Import-Csv "E:\cryptoML-walkforward-results\p2_h1_individual_full_v1.csv" |
  Select-Object feature, variant, folds, pooled_accuracy, mean_fold_delta_vs_ema2_points, positive_folds, negative_folds |
  Format-Table -AutoSize
\`\`\`

La colonne \`mean_fold_delta_vs_ema2_points\` mesure l'avantage ou le désavantage de l'indicateur sur **les mêmes closes H1 non plates**, par rapport à notre LightGBM EMA2 figé. Les tests sur les checks déjà inspectés, ainsi que l'essai de 60 variantes, empêchent toute affirmation de puissance prédictive indépendante ou rentabilité.
