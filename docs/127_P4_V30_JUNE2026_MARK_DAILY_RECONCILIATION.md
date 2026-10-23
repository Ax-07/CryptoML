# V30 — Protocole gelé : comparaison archive mensuelle / archive quotidienne mark BTC 2026-06

## Constat préalable
V24 : 80 partitions vérifiées sur 84, quatre absentes, aucune `FAILED_INTEGRITY`. V29 : archive mensuelle `markPriceKlines` de BTCUSDT juin 2026 conforme à son CHECKSUM Binance mais **8352 M5 sur 8640** ; 288 timestamps absents ; première absence le 2026-06-29 à 00:00 UTC. Cette absence pourrait constituer une journée entière, sans preuve sur la fin avant comparaison exhaustive. Ne pas retenter le même lot V27.

## Test V30 strict, sans écriture marché
Télécharger **uniquement** deux ZIPs officiels et leurs CHECKSUMs : `futures/um/monthly/markPriceKlines/BTCUSDT/5m/BTCUSDT-5m-2026-06.zip` et `futures/um/daily/markPriceKlines/BTCUSDT/5m/BTCUSDT-5m-2026-06-29.zip`. Contrôler les SHA256 *avant* décodage, vérifier ZIP unique et limites de décompression, analyser le décalage exact des 288 timestamps, vérifier par le parseur V23 chacune des 29 journées mensuelles présentes et les 288 barres de l'archive quotidienne. Refuser si chevauchement, dates hors mois, unité temporelle différente, trous résiduels, forme ou géométrie invalide.

Si la journée quotidienne vérifiée réconcilie la grille temporelle, le rapport conserve le statut **SOURCE_RECONCILIATION_CANDIDATE_NOT_PUBLISHED**. Cela n'autorise pas à publier, fusionner, imputer ou promouvoir une partition. Il faudra décider d'une adaptation d'ingestion explicite, testée, avec preuve de provenance par archive et nouveau protocole. Éviter tout PnL, entraînement ou prétention OOS.

## Commandes PowerShell

```powershell
Set-Location "E:\cryptoML-controlled-batches-v27"
git fetch origin
git worktree add "E:\cryptoML-daily-mark-v30" origin/research/btc-june2026-daily-mark-reconciliation-v30
Set-Location "E:\cryptoML-daily-mark-v30\backend"
$python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"

& $python -m pytest tests/test_funding_basis_daily_mark_reconciliation_v30.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V30" }

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }

$report = "E:\cryptoML-walkforward-results\p4_v30_btc_2026_06_daily_mark_reconciliation_v1.json"
if (Test-Path $report) { throw "STOP : rapport déjà existant" }
& $python -m mlpipeline.funding_basis_daily_mark_reconciliation_v30 `
    --permit-downloads `
    --output-json $report
if ($LASTEXITCODE -ne 0) { throw "STOP : réconciliation V30" }
Get-Content $report -Raw
```

Les fichiers des tentatives V27 et du rapport V29 ne doivent jamais être supprimés ou écrasés.
