# V13 — Recensement des signaux cinq stratégies, sans labels

Branche : \`research/multistrategy-candidate-census-v13\`. Voir [protocole gelé](95_P4_V13_FROZEN_MULTISTRATEGY_CENSUS_PROTOCOL.md). Pas de surveillance Binance et aucun ordre.

## Installation Windows

\`\`\`powershell
cd E:\cryptoML-paper-observer-v10
git fetch origin
git worktree add E:\cryptoML-multistrategy-v13 origin/research/multistrategy-candidate-census-v13
cd E:\cryptoML-multistrategy-v13\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_multistrategy_census_v13.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Exécuter le recensement seulement si les deux séries de tests réussissent.** Les résultats ne sont pas un backtest et ne démontrent aucun avantage. Attention : V9 utilise une boucle d'analyse sur les M5 et le calcul peut être relativement long sur les 21 mois.

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.multistrategy_census_v13 \`
  --data-root "E:\cryptoML\data" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v13_multistrategy_census_v1.json" \`
  --output-signals-csv "E:\cryptoML-walkforward-results\p4_v13_multistrategy_signals_v1.csv" \`
  --output-monthly-csv "E:\cryptoML-walkforward-results\p4_v13_multistrategy_monthly_v1.csv"
\`\`\`

**Les trois fichiers doivent être nouveaux et situés hors P0.** Ils comportent (1) JSON de référence/compte des cinq mécanismes, (2) chaque signal causal horodaté sans outcome, (3) les comptes mois×actif×direction×stratégie. Il est normal que certaines stratégies produisent peu, voire zéro signal.

## Analyse ultérieure, non lancée

L'étape suivante V14 fixera un évaluateur commun : pas d'entrée avant signal, frictions, capital/conflits, sorties et horizon prescrit par timeframe, contrôle du taux d'occupation et comparaison au zéro trade. Entraîner une IA de choix avant de connaître les réalités économiques et les coûts serait prématuré.

**Les périodes P0 déjà inspectées ne peuvent pas servir de holdout réellement indépendant pour décider de trader.** Les modèles testés sur ces données exigent une validation supplémentaire sur de nouvelles données verrouillées à l'avance.
