# V14 — Comparatif historique des cinq stratégies : exécution Windows

Lire d'abord [97_P4_V14_FROZEN_MULTISTRATEGY_ECONOMIC_PROTOCOL.md](97_P4_V14_FROZEN_MULTISTRATEGY_ECONOMIC_PROTOCOL.md). **V13 est validée** sur 6 211 signaux par l'opérateur ; V14 est un nouveau module à vérifier. **Ne pas utiliser le trading en direct.**

\`\`\`powershell
cd E:\cryptoML-multistrategy-v13
git fetch origin
git worktree add E:\cryptoML-multistrategy-v14 origin/research/multistrategy-economic-comparison-v14
cd E:\cryptoML-multistrategy-v14\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_multistrategy_economics_v14.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Ne lancer le calcul réel que si les deux suites de tests passent** :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.multistrategy_economics_v14 \`
  --data-root "E:\cryptoML\data" \`
  --reference-json "E:\cryptoML-walkforward-results\p4_v13_multistrategy_census_v1.json" \`
  --reference-signals-csv "E:\cryptoML-walkforward-results\p4_v13_multistrategy_signals_v1.csv" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v14_multistrategy_economics_v1.json" \`
  --output-events-csv "E:\cryptoML-walkforward-results\p4_v14_multistrategy_events_v1.csv" \`
  --output-monthly-csv "E:\cryptoML-walkforward-results\p4_v14_multistrategy_monthly_v1.csv"
\`\`\`

Tous les outputs doivent être nouveaux, hors P0. Le module refuse de calculer si l'empreinte P0 et chaque événement V13 recomputé ne correspondent pas aux références V13. Le test recherche respecte les 84 Parquets vérifiés et n'ouvre ni ADA ni juillet–septembre 2026. Certains signaux sont exclus faute de fenêtre de maturité complète.

Les montants sont des **markouts théoriques individuels**, pas le rendement d'un portefeuille capitalisé. Les hypothèses de frais 2/5/10 bp ne reproduisent ni spread ni slippage/funding et peuvent être insuffisantes. Des fenêtres peuvent se chevaucher. Les horizons H4/H1/M5 ne sont pas directement comparables et il est interdit de sélectionner un gagnant à partir d'un résultat sur le passé déjà observé. Étape suivante seulement après validation : diagnostic des résultats et conception de la politique de méta-sélection, y compris abstention.
