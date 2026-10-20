# P2 — Diagnostic économique H1 hors ligne, sans exécution : commandes

**Usage :** étude post hoc sur les checks P2 historiques déjà explorés. Aucun nouveau fit, aucune connexion Binance, aucun accès P0, ADA, OOS de fold, final90, P3. **Pas de test 72h**. Ce programme traite uniquement les résultats déjà écrits :
- \`p2_h1_binary_full_v1.json\`
- \`p2_h1_binary_full_predictions_v1.parquet\`
- \`p2_feature_reduction_individual_remaining25_full_v1.json\`
- \`p2_ema9_26_h1h6_full_v1.json\`
- \`p2_ema9_26_h1h6_full_predictions_v1.parquet\`

**Attention fondamentale : ce n'est PAS un backtest de trading exécutable.** La métrique monétaire est un *markout hypothétique* de close(t) à close(t+5m), avec une direction disponible conceptuellement après close(t). Une opération ne pourrait pas être garantie à close(t). Impossible d'estimer des exécutions réelles, un spread historique, un slippage ou le PnL sur ces seules données. Les coûts 0/2/5/10 bps sont des sensibilités de frais aller-retour hypothétiques, pas des taux mesurés. Les observations plates sont comptées dans le dénominateur économique (et supportent les coûts), mais exclues uniquement de l'accuracy directionnelle. Il ne faut pas additionner les rendements d'actifs simultanés pour prétendre à un capital ou un portefeuille ; ni extrapoler des performances futures.

## Étape 1 — Tests

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h1_chronological_markout.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

En cas d'échec, ne pas lancer le rapport et renvoyer la trace.

## Étape 2 — Rapport complet, sans entraînement ni REST

Les deux noms de sortie ci-dessous doivent ne pas exister :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_chronological_markout \`
  --binary-report "E:\cryptoML-walkforward-results\p2_h1_binary_full_v1.json" \`
  --binary-predictions "E:\cryptoML-walkforward-results\p2_h1_binary_full_predictions_v1.parquet" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" \`
  --h1h6-predictions "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" \`
  --output-json "E:\cryptoML-walkforward-results\p2_h1_markout_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p2_h1_markout_v1.csv"
\`\`\`

## Étape 3 — Lire les résultats

\`\`\`powershell
Import-Csv "E:\cryptoML-walkforward-results\p2_h1_markout_v1.csv" |
  Select-Object model, roundtrip_cost_bps, nonflat_accuracy,
    gross_proxy_mean_bps, net_proxy_mean_bps |
  Format-Table -AutoSize

$r = Get-Content "E:\cryptoML-walkforward-results\p2_h1_markout_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.time_replay | Format-List
$r.fold_differences | Format-List
$r.pooled.paired | Format-List
\`\`\`

Les scores sont produits pour les **8 modèles/règles** déjà sauvegardés. Les résultats par actif et par fold sont intégralement conservés dans le JSON, ainsi que quatre bandes de confiance prédéfinies pour chaque modèle binaire. On ne retient pas une bande de confiance, un seuil de frais ou un crypto-actif gagnant pour annoncer ensuite une preuve sur les mêmes checks.

## Suite logique

Si les tests passent, interpréter le markout idéal et les sensibilités, puis préparer un **simulateur réellement exécutable avec prix d'entrée postérieurs à la prédiction**, notamment avec données bid/ask et latence horodatées si elles existent. Ne pas transformer ce rapport en affirmation de rentabilité ou de signal validé.
