# 10 — Environnement de développement, configuration et exploitation

> **Attention :** ces commandes décrivent une **cible à implémenter**. Le ZIP de documentation **ne contient pas** les modules Python/Next.js ni un `docker-compose.yml` fonctionnel. Adapter à la structure réelle de ton dépôt VS Code.

## Versions et outils (à figer à l'implémentation)

- Node.js LTS compatible avec la version de Next.js installée, TypeScript, gestionnaire npm/pnpm existant.
- Python 3.11+ sous environnement virtuel ; gestion des dépendances via `pyproject.toml` (uv/Poetry ou pip, choix à documenter).
- FastAPI, Uvicorn, Pydantic, Pandas, NumPy, PyArrow, DuckDB, scikit-learn, LightGBM, MLflow, SHAP, pytest.
- PostgreSQL si l'application propose déjà des expériences et données API persistantes.

Ne pas inscrire de numéros de version arbitraires : verrouiller les versions réellement compatibles dans les lockfiles et enregistrer `pip freeze`/commit par run.

## Plan de lancement local prévu

```bash
# Lorsque le backend et ses dépendances auront été créés :
cd backend
python -m venv .venv
# activation selon ton système
pip install -e '.[dev]'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Dans un autre terminal, lorsque le frontend existe :
cd frontend    # ou la racine du Next.js existant
npm install
npm run dev
```

API cible : `http://localhost:8000/docs` ; Next.js : `http://localhost:3000`. Si le projet existant utilise `src/app`, `pnpm` ou des chemins différents, conserver ses conventions.

## Configuration

Modèle fourni à la racine : `.env.cryptoml.example`. Ajouter les valeurs à l'environnement du backend, jamais dans des composants frontend :

- `DATABASE_URL` : SQL local (seulement si DB activée).
- `DATA_ROOT` : répertoire local de Parquet.
- `ARTIFACTS_ROOT` : modèles/rapports.
- `MLFLOW_TRACKING_URI` : tracking local.
- `CORS_ORIGINS` : origins autorisées, par exemple `http://localhost:3000`.
- `BINANCE_FAPI_BASE_URL` : endpoint de marché public ; pas de clé privée d'exchange.

Variables préfixées `NEXT_PUBLIC_` sont **publiques dans le navigateur** ; ne jamais y stocker des secrets.

## Observabilité

- Logs JSON contenant `request_id`, `run_id`, `dataset_id` ; horodatages UTC.
- Métriques ingestion (bougies, erreurs, délais) et jobs (statut, durée, échecs).
- Une prévision doit être identifiable depuis le replay jusqu'à la version et au fold du modèle.

## Sécurité et contraintes

- Ne jamais exposer un serveur de développement sans contrôle d'accès.
- En production, authentifier `training-runs` et protéger ressources lourdes par quotas/rate-limit.
- Vérifier chemins de fichiers : empêcher path traversal lors d'accès aux artefacts.
- Les données d'exchange utilisées ici sont publiques ; ne pas demander de credentials de trading.
- Pas d'ordre live. Toute extension future exigerait une revue sécurité/risk management séparée.

## Première démonstration réussie

1. 5 paires M5 chargées sans anomalies bloquantes.
2. 1 modèle H6 entraîné sur historique propre et métriques OOS persistées.
3. `/health`, `/market/candles`, `/evaluations/{id}`, `/replay` fonctionnels.
4. La page « Qualité prédictive » affiche les vraies données, avec badges `OOS`, périodes, dates et sample count ; le mode replay ne voit pas le futur.
