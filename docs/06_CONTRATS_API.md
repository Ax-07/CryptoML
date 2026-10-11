# 06 — API FastAPI : ressources et conventions

**Statut : contrat cible à implémenter.** Source machine lisible : [`../contracts/openapi.yaml`](../contracts/openapi.yaml). Base locale proposée : `http://localhost:8000/api/v1` ; docs FastAPI : `/docs`.

## Conventions

- JSON et UTF-8 ; `snake_case` ; timestamps UTC ISO-8601 avec `Z` ; horizon exprimé en **nombre de bougies**.
- `symbol` au format fournisseur (`SOLUSDT`), `market=usdt_perpetual`, `timeframe=5m`.
- Réponses paginées `items`, `next_cursor`, `total` facultatif. Tri par défaut ascendant pour chandeliers, décroissant pour prédictions.
- Chaque prédiction historique a un `model_version`, un `run_id`, `feature_schema_version`, `bar_close_at`, `prediction_issued_at`, `label_mature_at`.
- Statut vérité : `PENDING` / `MATURED` / `UNAVAILABLE` ; `actual_class=null` tant que verdict indisponible.
- Erreurs : JSON `{ "error": { "code": "...", "message": "...", "request_id": "..." } }` ; bons codes HTTP 400, 404, 409, 422, 429, 500.
- Les endpoints `POST` d'entraînement/évaluation sont réservés aux utilisateurs authentifiés une fois l'application exposée sur réseau.

## Endpoints recommandés

| Méthode | Route | Réponse / but |
|---|---|---|
| GET | `/health` | disponibilité basique |
| GET | `/api/v1/markets/symbols` | liste des paires éligibles et couverture |
| GET | `/api/v1/market/candles` | OHLCV pour Lightweight Charts |
| GET | `/api/v1/datasets` | datasets et qualité |
| GET | `/api/v1/datasets/{dataset_id}/quality` | trous, volumes, couverture |
| POST | `/api/v1/training-runs` | créer un run, renvoyer 202 + `run_id` |
| GET | `/api/v1/training-runs/{run_id}` | statut/progression/artefacts |
| GET | `/api/v1/models` | modèles versionnés par horizon |
| GET | `/api/v1/evaluations/{evaluation_id}` | métriques, confusion, calibration, breakdown |
| GET | `/api/v1/predictions` | filtrer historique OOS ou inférence |
| GET | `/api/v1/predictions/latest` | dernière prédiction d'un symbole/horizon/modèle |
| GET | `/api/v1/replay` | chandeliers jusqu'au curseur, prédictions éligibles, verdicts matures |
| GET | `/api/v1/explanations/{prediction_id}` | explications SHAP stockées |

### Requête d'entraînement

```http
POST /api/v1/training-runs
Content-Type: application/json
```

```json
{
  "dataset_id": "ds_pilot_001",
  "feature_schema_version": "features_v1",
  "label_config_id": "vol_scaled_v1",
  "horizons": [3, 6, 12],
  "model_family": "lightgbm",
  "validation_protocol": "purged_rolling_walk_forward_v1"
}
```

Réponse `202` :

```json
{ "run_id": "run_20261012_001", "status": "QUEUED" }
```

### Réponse de prédiction

```json
{
  "prediction_id": "pred_001",
  "symbol": "SOLUSDT",
  "market": "usdt_perpetual",
  "timeframe": "5m",
  "bar_close_at": "2026-08-01T12:00:00Z",
  "prediction_issued_at": "2026-08-01T12:00:02Z",
  "model_version": "lgbm_h6_v1",
  "run_id": "run_001",
  "feature_schema_version": "features_v1",
  "horizon_bars": 6,
  "probabilities": {"up": 0.62, "neutral": 0.21, "down": 0.17},
  "predicted_class": "UP",
  "label_mature_at": "2026-08-01T12:30:00Z",
  "evaluation_status": "PENDING",
  "actual_class": null,
  "actual_return": null,
  "demo": true
}
```

Données **purement fictives** ; `demo=true` signifie que cette sortie ne doit jamais être agrégée comme une mesure réelle. Pour des événements historiques de walk-forward, `prediction_issued_at` représente le timestamp **simulé** d'émission ; conserver séparément le moment où le résultat a été produit par le pipeline de recherche si utile.

## Détails de `GET /api/v1/market/candles`

Paramètres requis : `symbol`, `from`, `to` (UTC), optionnels `timeframe=5m`, `limit`. Retour : `[{time,open,high,low,close,volume}]` avec `time` en **secondes UNIX de début de bougie** attendu par Lightweight Charts ; sinon utiliser le contrat `bar_open_at` + adaptateur frontend explicite. Ne pas mélanger seconde et milliseconde.

## Détails de `GET /api/v1/replay`

Paramètres : `evaluation_id`, `symbol`, `horizon_bars`, `cursor_at`, fenêtre visible et limite. Le serveur applique :

- pas de bougie dont `bar_close_at > cursor_at` ;
- pas de prédiction dont `prediction_issued_at > cursor_at` ;
- pas de vérité terrain si `label_mature_at > cursor_at` ;
- ne montrer que les prédictions appartenant à l'évaluation OOS sélectionnée.

Ce filtrage doit être **côté serveur**, pas seulement masqué en CSS/JS.

## Évolutions (hors MVP)

WebSocket/SSE pour statut d'entraînement, génération périodique de prédictions, gestion des tests de stratégie/transactions, calcul d'incertitude en ligne. Ne pas implémenter d'API de trading live dans cette phase.
