# 02 — Sources de données, schémas et contrôles qualité

## Marché et sources

- Contrats perpétuels **Binance USDⓈ-M**, paires `BTCUSDT`, `ETHUSDT`, etc.
- Archives historiques : https://data.binance.vision/ et dépôt descriptif https://github.com/binance/binance-public-data.
- Complément récent : `GET https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m...`.
- Le *funding*, *mark price* et *open interest* sont réservés à une phase ultérieure, sauf disponibilité historique et horodatage fiable ; **ne pas remplir les trous par des valeurs inventées**.

## Import initial

1. Pilote : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT ; fenêtre historique commune maximale réaliste.
2. Archives mensuelles puis quotidiennes ; vérification des CHECKSUM si proposés.
3. Décompresser et normaliser le schéma ; réconcilier doublons archive/API avec règles documentées.
4. Exclure les bougies non clôturées au moment de l'import/inférence.
5. Identifier trous, timestamps incohérents et données hors marché ; publier rapport qualité.
6. Stocker données **brutes immuables** puis version propre Parquet ; garder checksums/source pour traçabilité.
7. Étendre à 20–30 paires sélectionnées avec liquidité minimale documentée, sans biaiser l'évaluation vers seules paires encore vivantes aujourd'hui.

## Schéma canonique des bougies M5

| Champ | Type | Commentaire |
|---|---|---|
| `exchange` | string | `binance` |
| `market` | string | `usdt_perpetual` |
| `symbol` | string | exemple `BTCUSDT` |
| `timeframe` | string | `5m` |
| `bar_open_at` | datetime UTC | début de la bougie |
| `bar_close_at` | datetime UTC | **borne exclusive** `bar_open_at + 5 minutes` |
| `open`, `high`, `low`, `close` | float positif | OHLC |
| `volume_base`, `volume_quote` | float non négatif | volumes dans les unités explicites |
| `trade_count` | entier | si disponible |
| `taker_buy_volume_base`, `taker_buy_volume_quote` | float | si disponible |
| `source` | string | `archive_monthly`, `archive_daily`, `rest` |
| `ingested_at` | datetime UTC | date import, pas date marché |
| `is_final` | boolean | bougie clôturée et confirmée |

**Convention cruciale** : Binance publie souvent `close_time = open_time + interval - 1 ms`. Dans notre base, `bar_close_at = open_time + interval` représente la **fin logique exclusive**, utilisée pour la disponibilité et le replay. Ne pas confondre ces deux conventions.

## Validations à appliquer par symbole

- Unicité `(exchange, market, symbol, timeframe, bar_open_at)`.
- Alignement exact aux pas de 300 secondes ; ordre strictement croissant.
- `low <= min(open, close)`, `high >= max(open, close)`, `low <= high`.
- Prix strictement positifs ; volumes et trade_count non négatifs.
- Détection des trous (`diff(bar_open_at) != 5min`), doublons, volumes nuls suspects, pics aberrants, changements de listing.
- En cas de trous : marquer période comme inéligible si une fenêtre de features ou de label la traverse ; **pas de forward-fill des bougies OHLC inventées**.
- Documenter les effets de contrats suspendus/délistés et survivorship bias.
- Séries contextuelles BTC/ETH alignées sur la même bougie ; si absentes, l'échantillon concerné est rejeté ou traité par une politique de valeurs manquantes fixée a priori.

## Stockage proposé

```text
data/raw/binance/usdt_perpetual/5m/BTCUSDT/2025-01.zip
data/processed/ohlcv/exchange=binance/market=usdt_perpetual/
    timeframe=5m/symbol=BTCUSDT/year=2025/month=01/part-000.parquet
```

Pour chaque snapshot du dataset, produire un manifest : `dataset_id`, date de construction, intervalle UTC, liste des symboles, schéma, contrôles qualité, hashes et version du code.

## Informations disponibles « point-in-time »

Les données peuvent être connues après leur date de marché. Pour la simulation, vérifier `available_at <= prediction_issued_at`. Ne jamais utiliser un indicateur recalculé a posteriori si sa version n'était pas disponible lors de la prédiction historique. Le MVP peut se limiter à OHLCV clôturé (source moins ambiguë) et enregistrer la latence supposée.

## Hors périmètre de cette phase

Prix de liquidation, exécution, carnet d'ordres historique point-in-time et funding exact par position. Ils deviendront requis pour un futur backtest perpétuel **de trading**, pas pour l'évaluation directionnelle actuelle.
