# V16 — Audit volume / taker et échantillon funding (lecture seule)

Voir le protocole de recherche préenregistré : [100_P4_V16_VOLUME_TAKER_FUNDING_DATA_PROTOCOL.md](100_P4_V16_VOLUME_TAKER_FUNDING_DATA_PROTOCOL.md).

Le module réutilise la validation des 84 partitions P0 existantes et relit les champs historiques volume/taker sans écrire dans P0. Il produit un **recensement sans résultats économiques**, sans trading et sans machine learning. La sonde Binance historique est **facultative** et ne requiert pas de clé.

## Installation

```powershell
cd E:\cryptoML-multistrategy-v14
git fetch origin
git worktree add E:\cryptoML-volume-funding-v16 origin/research/volume-funding-readiness-v16
cd E:\cryptoML-volume-funding-v16\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_volume_funding_readiness_v16.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Si un test échoue, ne pas lancer l'audit historique** et transmettre la traceback. Les tests sont synthétiques et n'ont pas été exécutés depuis GitHub sur la machine de l'opérateur.

## Audit historique hors réseau

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.volume_funding_readiness_v16 `  
  --data-root "E:\cryptoML\data" `  
  --output-json "E:\cryptoML-walkforward-results\p4_v16_volume_readiness_v1.json" `  
  --output-csv "E:\cryptoML-walkforward-results\p4_v16_a06_signals_v1.csv"
```

Les sorties doivent être **nouvelles** et hors du P0. Pour éviter de dépendre d'espaces après backticks lors du copier-coller, il est également possible d'écrire la commande sur une seule ligne.

## Sonde historique de funding (facultative)

Après succès du mode hors réseau, exécuter le même programme avec **de nouveaux noms** de sortie et ajouter `--probe-funding-history`. Cela effectue quatre GET Binance public `/fapi/v1/fundingRate` ciblant 1er–3 juin 2025. Les statuts `HISTORICAL_SAMPLE_RETRIEVED`, `NO_SAMPLE_FOUND` ou `UNVERIFIED` sont sauvegardés dans le JSON. **Même si un taux historique est présent, la couverture sur 21 mois n'est pas établie**, et il ne s'agit pas d'un arbitrage rentable prouvé. L'échec 403/451 ou timeout ne doit pas être contourné.

## Interprétation et suite

1. Vérifier l'absence d'anomalies dans les volumes taker (sinon arrêt).
2. Comparer le nombre de signaux A06 LONG/SHORT par actif et la densité dans le temps, **sans consulter encore leurs performances**.
3. Geler un modèle de frictions/coûts et un horizon de sortie A06 avant de mesurer des rendements.
4. Pour le carry funding / basis, préparer un jeu spot+perp/funding réellement synchronisé, un modèle des deux jambes, funding réel et coût/risque de liquidation. **Ne pas traiter le mark comme spot exécutable**.

Aucun lancement de collecte en direct, aucun ordre, aucun entraînement.
