# CryptoML — P0 Backend (USD-M Futures 5m)

Première livraison **exécutable** : ingestion de bougies Binance USDⓈ-M perpétuelles (M5), contrôle de qualité, stockage Parquet et API de lecture FastAPI. **Aucun ordre n'est envoyé et aucune clé Binance n'est nécessaire.**

## Prérequis

- Python 3.11+ ; connexion Internet pour l'installation et les téléchargements Binance.
- Extraire le ZIP **à la racine** du dépôt `CryptoML`. Le dossier GitHub `docs/` est conservé intact.

## Installation (VS Code / PowerShell)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Sur Linux/macOS :

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Si l'activation de PowerShell est bloquée, utiliser directement `.venv\Scripts\python.exe -m pip install -e ".[dev]"` et `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`.

## Démarrer l'API (depuis `backend/`)

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Santé : <http://127.0.0.1:8000/health>
- OpenAPI/Swagger : <http://127.0.0.1:8000/docs>
- Paires chargées : <http://127.0.0.1:8000/api/v1/markets/symbols>
- Inventaire : <http://127.0.0.1:8000/api/v1/datasets>

Avant un premier import, l'inventaire et les bougies sont vides : **c'est normal**.

## Importer la première paire (archive officielle, checksum vérifié)

```bash
python -m ingestion.cli import-month --symbol BTCUSDT --month 2026-08
python -m ingestion.cli list-data
```

Importer les **cinq paires pilote** sur les deux derniers mois *complets* (BTC, ETH, SOL, XRP, ADA) :

```bash
python -m ingestion.cli import-pilot --months 2
```

Selon la disponibilité des archives mensuelles Binance, certains mois récents peuvent ne pas encore être publiés. La commande signale les archives manquantes. Utilise les archives pour le fond historique ; évite de lancer des centaines de requêtes REST pour deux années complètes.

Pour compléter une plage récente (maximum 31 jours par appel ; `--end` exclusif) :

```bash
python -m ingestion.cli sync-rest --symbol BTCUSDT --start 2026-10-01T00:00:00Z --end 2026-10-02T00:00:00Z
```

Ces commandes ont besoin d'un accès à `data.binance.vision` ou `fapi.binance.com`. Certains pays, réseaux d'entreprise ou proxys peuvent les bloquer. Une erreur réseau est signalée ; aucun contenu factice n'est généré.

## Interroger les chandeliers depuis Next.js

```text
GET http://127.0.0.1:8000/api/v1/market/candles?symbol=BTCUSDT&from=2026-08-01T00:00:00Z&to=2026-08-02T00:00:00Z&limit=500
```

Réponse : tableau `{time, open, high, low, close, volume}`. `time` est le début de la bougie, en **secondes UNIX**, conforme à Lightweight Charts. Limite : 1 500 bougies, fenêtre <= 31 jours, ordre croissant. Si plus de 1 500 bougies : répéter avec `from` égal au **temps d'ouverture de la dernière bougie + 5 minutes**.

Autres routes : `GET /api/v1/datasets/{dataset_id}/quality`, `GET /health`. Le catalogue de symboles est celui des fichiers **effectivement importés**, pas l'univers Binance exhaustif.

## Fichiers écrits

```text
data/
  raw/binance/usdt_perpetual/5m/BTCUSDT/2026-08.zip
  raw/binance/usdt_perpetual/5m/BTCUSDT/2026-08.zip.CHECKSUM
  processed/ohlcv/exchange=binance/market=usdt_perpetual/
    timeframe=5m/symbol=BTCUSDT/year=2026/month=08/
      part-000.parquet
      manifest.json
      quality_report.json
```

Le chemin est relatif à la racine du dépôt par défaut ; variable d'environnement `DATA_ROOT` pour le modifier (préférer un chemin absolu). `data/` est ignoré par Git. Les ZIP bruts sont conservés immuables : si Binance les corrige plus tard, le pipeline signale la différence au lieu de réécrire silencieusement les données.

Un trou interne est documenté, **pas comblé artificiellement**. L'absence de données avant le listing ou après un delisting exige une interprétation distincte du nombre de trous internes. La source REST n'est pas un substitut au contrôle de qualité du dataset.

## Tests

```bash
pytest
```

Tests unitaires offline : schéma/timestamps M5, exclusion de la bougie ouverte, trous, SHA256 des archives, pagination REST et API, idempotence/conflits d'import. Des données fictives sont utilisées **uniquement comme fixtures de test** et ne sont jamais enregistrées comme données de marché opérationnelles.

## Limites assumées de P0

- **Pas encore d'entraînement LightGBM** : livraison P1.
- Pas encore d'API de replay ou de prédictions : livraison après entraînement.
- Pas de WebSockets, jobs programmés, base PostgreSQL ou connexion exchange privée.
- Les données historiques sont **OHLCV exchange** ; elles ne démontrent ni fraîcheur d'un signal temps réel ni capacité à passer un ordre.
- Pas de certification d'éligibilité d'une paire par l'API `exchangeInfo` : privilégier les paires `USDT` perpétuelles Binance connues du périmètre pilote. Cette vérification devra être ajoutée avant la généralisation de l'univers.
- Ce prototype local n'a pas d'authentification : ne pas exposer sur Internet tel quel.

## Sources techniques

- [Archives publiques Binance et CHECKSUM](https://github.com/binance/binance-public-data)
- [Klines Binance Futures USD-M](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
- Documentation interne préexistante : `../docs/`.
