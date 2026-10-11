# CryptoML — Checklist de réalisation

**À importer dans les issues ou à cocher depuis VS Code.** Le statut initial est *non implémenté*.

## Sprint 0 — Structure et data
- [ ] Examiner le projet existant ; ajouter `backend/` sans casser Next.js.
- [ ] Ajouter FastAPI `/health` + tests `pytest`.
- [ ] Downloader archives Binance USD-M Futures M5, 5 symboles pilote.
- [ ] Mettre en place contrôle timestamps/duplicates/gaps/OHLC ; rapport par symbole.
- [ ] Stocker Parquet + manifest de dataset versionné.

## Sprint 1 — Première preuve de qualité prédictive
- [ ] Implémenter features causales et tests anti-fuite.
- [ ] Implémenter labels H6 avec seuil de volatilité passée et `label_mature_at`.
- [ ] Split temporel purgé ; baseline puis LightGBM global.
- [ ] Générer confusion, macro-F1, balanced accuracy, log loss et calibration OOS.
- [ ] Sauvegarder model/metrics/predictions OOS et versions.

## Sprint 2 — API + Replay
- [ ] Définir Pydantic avec OpenAPI.
- [ ] Endpoints candles, dataset, evaluations, predictions, replay.
- [ ] Connecter Next.js shadcn/ui et Lightweight Charts.
- [ ] Implémenter contrôles Play/Pause/Step et filtres.
- [ ] Tests backend masquage du futur + E2E replay.

## Sprint 3 — Extension recherche
- [ ] Horizons H3/H12, univers 20–30 paires éligibles.
- [ ] Walk-forward à plusieurs folds + tests symboles inédites.
- [ ] Explicabilité, expériences comparées, MLflow.
- [ ] Inférence après clôture confirmée et résolution différée des labels.

## Hors périmètre pour le moment
- [ ] Backtesting LONG/SHORT avec financement/frais.
- [ ] Exécution d'ordres réels — à reconsidérer seulement après revue complète du risque.
