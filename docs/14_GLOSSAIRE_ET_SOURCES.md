# 14 — Glossaire et références techniques

## Glossaire

- **M5** : bougie de cinq minutes.
- **USDⓈ-M Perpetual** : contrat à terme perpétuel dont la marge/les règlements sont liés à des actifs de type USDT/USDC selon le contrat ; ici priorité aux paires USDT.
- **OHLCV** : ouverture, plus haut, plus bas, clôture, volume.
- **Feature** : donnée d'entrée disponible lors de la prédiction.
- **Label** : réponse cible déterminée à l'aide d'un résultat futur, réservée au train/évaluation.
- **Horizon H6** : six bougies M5 futures, soit 30 minutes.
- **OOS** (*out of sample*) : période et/ou instrument non utilisé pour entraîner le modèle testé.
- **Walk-forward** : répétition entraînement puis test sur une fenêtre temporelle ultérieure.
- **Purging** : exclusion des exemples dont l'horizon de label déborde sur la fenêtre suivante.
- **Embargo** : intervalle additionnel sans échantillons entre deux segments.
- **Holdout symbole** : crypto complètement exclue de la phase d'apprentissage/réglage.
- **Calibration** : accord entre probabilité annoncée et fréquence réelle de réalisation.
- **Balanced Accuracy** : rappel moyen des classes présentes.
- **SHAP** : explication de la contribution de features à une prédiction ; ce n'est pas une preuve de causalité.
- **Replay point-in-time** : visualisation qui ne révèle que les données connaissables à l'instant simulé.
- **Paires inédites** : jamais utilisées pour entraîner ni choisir paramètres du modèle final.
- **Funding** : paiements périodiques entre positions sur contrats perpétuels, hors cible directionnelle actuelle.

## Références officielles (vérifiées lors de la rédaction)

- Binance Public Data — https://github.com/binance/binance-public-data
- Binance API USDⓈ-M Futures (klines, funding, mark price, open interest) — https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
- Binance catalogue REST Futures — https://developers.binance.info/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data
- LightGBM documentation — https://lightgbm.readthedocs.io/en/stable/
- Scikit-learn TimeSeriesSplit — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- FastAPI — https://fastapi.tiangolo.com/
- MLflow tracking — https://mlflow.org/docs/latest/tracking/
- Next.js App Router — https://nextjs.org/docs/app
- shadcn/ui — https://ui.shadcn.com/docs/installation
- TradingView Lightweight Charts — https://tradingview.github.io/lightweight-charts/
- Apache ECharts — https://echarts.apache.org/en/index.html

Le schéma d'architecture reste valable même si les chemins/endpoints upstream changent. Vérifier la documentation de l'exchange lors de l'implémentation et ne pas déduire la disponibilité historique de données OI/funding d'un simple endpoint actuel.
