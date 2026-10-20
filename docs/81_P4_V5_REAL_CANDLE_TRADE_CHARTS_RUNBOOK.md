# P4 V5 — Graphiques M5 réels, positions backtest affichées clairement

**Bougies = données historiques Binance réelles et SHA-256 vérifiées. Positions = signaux et fills simulés par le backtest, PAS des ordres exécutés.** Les modèles ne passent aucun ordre live pendant fit/test.

Deux modes :
- `--mode model` : recrée **exactement** les décisions C0 Ridge9 (train déc. 2025–mars 2026, seuil scores B avril, diagnostics mai–juin 2026), en attend **84** et en visualise jusqu'à six (meilleurs, pires, timeout, médian). Les autres modèles C1/C2 ont pris 0 position ; aucune position C1/C2 n'existe à représenter.
- `--mode training` : montre six exemples de trades mécaniques B ayant servi à l'entraînement 4 mois de C0. Ils ont des issues historiques calculées pour les labels, mais ne sont **pas** des positions 'prises par le modèle' après fit.

Le visualiseur utilise le chargeur 21 mois vérifié et le calculateur d'événements existant, rejoue les sorties fixes sur OHLC pour retrouver le **bar_close_at de la sortie**, dessine de vraies chandelles, stop/target, signal, entrée et sortie. N'utilise ni matplotlib ni données internet dans le navigateur : fichier **HTML autonome SVG**.

## Windows

```powershell
cd E:\cryptoML-pullback-economics-v4
git fetch origin
git worktree add E:\cryptoML-pullback-trade-charts-v5 origin/research/pullback-trade-charts-v5
cd E:\cryptoML-pullback-trade-charts-v5\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_trade_charts_v5.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

# Uniquement si les tests sont verts :
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_trade_charts_v5 `
  --data-root "E:\cryptoML\data" `
  --mode model `
  --output-html "E:\cryptoML-walkforward-results\p4_v5_ridge_c0_real_candles_model_trades_v1.html"

# Lancer en supplément si l'on veut regarder des exemples B de train :
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_trade_charts_v5 `
  --data-root "E:\cryptoML\data" `
  --mode training `
  --output-html "E:\cryptoML-walkforward-results\p4_v5_real_candles_B_training_examples_v1.html"

Start-Process "E:\cryptoML-walkforward-results\p4_v5_ridge_c0_real_candles_model_trades_v1.html"
```

Ne pas écraser un nom de sortie déjà présent. Les 6 charts sont **des illustrations choisies avec connaissance des labels** ; regarder des cas extrêmes ne constitue pas une estimation de performance. Les métriques correctes sont dans le rapport complet V3 ; le graphique n'est pas une politique de sélection.

## Interpréter

- La barre du signal `t` est connue après sa clôture.
- La bougie `t+1` permet la confirmation.
- L'entrée est l'OPEN de `t+2`, au même horodatage que le close de `t+1`. Cela ne signifie **pas** qu'une exécution réelle instantanée serait possible.
- Stop `entry−ATR14(t)`, target `entry+1.5 ATR14(t)`. La sortie est la première barrière rencontrée, ou la clôture de la sixième bougie ; si les deux barrières sont touchées, le stop est prioritaire. Cas gaps identiques au calculateur V1.
- Les lignes et les prix sont **hypothétiques** au niveau backtest, les OHLC proviennent des bougies Binance historiques.
