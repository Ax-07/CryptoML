# 09 — Stratégie de persistance et versionnement

## Trois catégories de données

1. **Parquet** : historique M5 brut/nettoyé, features et prédictions OOS volumineuses ; lecture analytique via DuckDB.
2. **PostgreSQL** : inventaire des datasets, modèles, expériences, splits, jobs, résultats agrégés, références de prédictions et horodatages ; API paginée.
3. **MLflow / artefacts** : modèle LightGBM, manifest d'entraînement, courbes de calibration, confusion, schéma des features et métriques détaillées.

Il est inutile de dupliquer plusieurs années d'OHLCV en Postgres pour le MVP si Parquet/DuckDB dessert correctement le replay. Les références SQL doivent contenir paths/checksums/version.

## Tables relationnelles proposées

| Table | Champs / clés principaux |
|---|---|
| `datasets` | `dataset_id`, `source`, `schema_version`, `start_at`, `end_at`, `checksum`, `created_at` |
| `dataset_symbols` | `(dataset_id, symbol)`, `coverage_start`, `coverage_end`, `missing_bars`, `is_eligible` |
| `feature_sets` | `feature_schema_version`, `schema_json`, `code_ref`, `created_at` |
| `label_configs` | `label_config_id`, `horizons`, `threshold_method`, `params_json` |
| `training_runs` | `run_id`, dataset_id, feature/version, label_config, params, status, started/finished |
| `validation_folds` | `fold_id`, `run_id`, time boundaries, purging, heldout_symbols |
| `model_versions` | `model_version`, `run_id`, `horizon_bars`, `artifact_uri`, `status` |
| `evaluations` | `evaluation_id`, `run_id`, `segment`, metrics, sample_count, start/end |
| `predictions` | `prediction_id`, `evaluation_id`, `fold_id`, `model_version`, `symbol`, `bar_close_at`, `prediction_issued_at`, `horizon_bars`, `up/neutral/down`, `predicted_class`, `label_mature_at`, `evaluation_status`, `actual_class`, `actual_return`, `demo` |
| `jobs` | `job_id`, `kind`, `run_id`, `status`, `progress`, `error_code`, `created_at`, `updated_at` |

## Index importants

- Bougies (Parquet) partitionnées par symbole/date, puis filtrées par range UTC.
- `predictions(symbol, horizon_bars, bar_close_at)`.
- `predictions(evaluation_id, bar_close_at)`.
- `training_runs(created_at desc)` et `model_versions(horizon_bars, status)`.
- Idempotence des prédictions : contrainte de cohérence `(evaluation_id, fold_id, symbol, horizon_bars, bar_close_at)` pour un résultat OOS donné, avec `model_version` correspondant.

## Distinction des dates

- `bar_open_at` et `bar_close_at` : événement de marché.
- `prediction_issued_at` : disponibilité simulée/réelle de la prévision.
- `label_mature_at` : date à laquelle la vérité peut être calculée.
- `generated_at` : instant où un batch de recherche a été calculé (peut être postérieur à la période simulée).
- `created_at` DB : date d'enregistrement de l'artefact.

Cette distinction évite de confondre **quand une donnée a été produite lors du backtest** et **quand elle aurait été connaissable sur le marché**.

## Gouvernance des modèles

Statuts proposés : `EXPERIMENTAL`, `VALIDATED_OOS`, `ARCHIVED`. Le passage à `VALIDATED_OOS` exige un rapport complet et ne signifie pas « profitable ». Stocker une référence immuable aux features, données, labels et code du run. Garder reproductibilité, tests d'intégrité et procédure de rollback.
