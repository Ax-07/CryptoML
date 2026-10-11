# 07 — Frontend Next.js / shadcn/ui : les 11 pages

Le wireframe V4 est la référence d'ergonomie. Le frontend cible : **Next.js App Router + TypeScript + shadcn/ui + Tailwind CSS**, graphiques **TradingView Lightweight Charts** pour prix/séries chronologiques et **ECharts** pour statistiques de modèle.

## Navigation et écrans

| N° | Page | Objectif et composants | Phase |
|---|---|---|---|
| 1 | **Vue d'ensemble** | KPIs de qualité prédictive OOS, tendances de score, couverture dataset, état modèle | MVP |
| 2 | **Qualité prédictive** | matrice de confusion, macro-F1, balanced accuracy, log loss, calibration, comparaison horizons, par crypto, **replay prédit vs réel sans trade** | MVP PRIORITAIRE |
| 3 | **Graphique & Trades** | chandeliers, marqueurs long/short/sorties, journal PnL, TP/SL | Futur : simulation de stratégie ; état explicite « à venir » |
| 4 | **Signaux** | prévisions UP/NEUTRAL/DOWN, horodatage, fraîcheur, historique et verdicts | MVP, sans conseil/ordre |
| 5 | **Interprétation** | importance globale des features, SHAP local, erreurs et régimes, effectifs | MVP après premiers runs |
| 6 | **Entraînement** | univers, dataset, features, labels, horizons, hyperparamètres, progression | MVP |
| 7 | **Walk-forward** | fenêtres train/val/test, purge, stabilité, holdouts paires inédites | MVP |
| 8 | **Données** | sources, couverture temporelle, trous, doublons, fraîcheur, qualité, symboles | MVP |
| 9 | **Expériences** | runs, versions, baselines, artefacts, comparaison par horizon | MVP |
| 10 | **Backtesting** | equity, PnL, frais, DD, comparaisons stratégies | Futur : non prioritaire ; afficher indisponible |
| 11 | **Paramètres** | exchange public, timeframe, timezone, chemins API et préférences UI | MVP |

**Principe produit** : la page « Qualité prédictive » décrit **les mouvements de marché**, tandis que « Graphique & Trades » et « Backtesting » concernent **des décisions de trading**, qui ne doivent pas être confondues.

## Composants UI réutilisables

```text
components/
  app-shell/sidebar
  global-filters (symbol, horizon, dataset, run, evaluation, segment)
  model-status-badge (training, OOS, demo, unseen-pair)
  metric-card / metric-trend / confidence-interval
  candle-chart / prediction-marker / synchronized-crosshair
  prediction-replay-controller (play, pause, step, speed, reset)
  prediction-inspector (probabilités, classe, maturation, étiquette)
  confusion-matrix / horizon-comparison / calibration-plots
  dataset-quality-grid / experiment-table
  loading-empty-error-states
```

## Filtres globaux

- Marché `usdt_perpetual` (défaut), exchange `binance`, timeframe `5m` (fixé dans le MVP).
- Symbole, horizon H3/H6/H12, période UTC, `model_version`, `run_id` et **segment d'évaluation**.
- Segment visible sur chaque graphique : train/val OOS/test final/paires inédites/démo. Ne pas agréger ces segments par défaut.
- Tous les écrans affichent clairement `Dernière donnée clôturée`, `Modèle`, `Horizon`, `Source`, `OOS vs train`.

## Graphiques

- **Lightweight Charts** : prix OHLC, sous-chart probabilités P(UP/NEUTRAL/DOWN), scores/charts temporels éventuellement synchronisés ; le replay pilote un unique `cursor_at`.
- **ECharts** : confusion, répartition des classes, bars horizon, heatmap symboles, calibration multi-classes, feature importance, SHAP.
- N'utiliser les marqueurs achat/vente **que dans les pages de trading futures** ; sur la page prédictive, marqueurs de **prévision** avec tri-classe.

## États et accessibilité

- États `loading`, `no-data`, `dataset-invalid`, `training-running`, `pending-label`, `model-not-ready`, `API-error`.
- Textes explicites (pas uniquement vert/rouge), navigation clavier, tooltips descriptifs, labels et contraste accessibles.
- Mobile : filtres en tiroir, graph occupent toute la largeur, panneaux inspecteur empilés sous le chart.
- Une métrique affichée doit mentionner son **échantillon, sa période et son segment**.

## API / frontend

Consommer un client typé dérivé de `contracts/openapi.yaml` si possible. Centraliser cache, erreurs, délais et fetch. Le frontend n'a pas accès aux données futures « brutes » en mode replay. Conserver l'identifiant sélectionné dans l'URL pour partager un état, sans y inclure de secret.
