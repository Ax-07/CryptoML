# V28 Runbook — diagnostic BTCUSDT spot 2026-06, SANS ingestion

Suite à l'échec du dernier plan V27 : journal `STARTED BTCUSDT 2026-06`, puis `FAILED ValueError Incorrect M5 count/shape` ; V24 indépendant donne 80 vérifiées, 4 manquantes, zéro `FAILED_INTEGRITY`.

Ne pas relancer `v19` ou `v19_resume` avant l'analyse. [Protocole gelé](124_P4_V28_BTC_JUNE2026_SPOT_DIAGNOSTIC_PROTOCOL.md). **Le code V28 télécharge uniquement la ZIP spot BTCUSDT juin 2026 et le CHECKSUM, sans écritures P0, V21, V23, sans publication de partition et sans trading.**

## Installation et tests

```powershell
cd E:\cryptoML-controlled-batches-v27
git fetch origin
git worktree add E:\cryptoML-spot-diagnostic-v28 origin/research/btc-june2026-spot-archive-diagnostic-v28
cd E:\cryptoML-spot-diagnostic-v28\backend

$python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_spot_diagnostic_v28.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V28 echoues" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests globaux echoues" }
```

## Diagnostic de lecture seule

```powershell
& $python -m mlpipeline.funding_basis_spot_diagnostic_v28 `
  --permit-downloads `
  --output-json "E:\cryptoML-walkforward-results\p4_v28_btc_2026_06_spot_diagnostic_v1.json"
if ($LASTEXITCODE -ne 0) { throw "STOP : diagnostic V28 echoue" }
Get-Content "E:\cryptoML-walkforward-results\p4_v28_btc_2026_06_spot_diagnostic_v1.json" -Raw
```

Le rapport expose `raw_csv_rows`, `data_rows`, `columns`, `expected_m5=8640`, `first_utc`, `last_utc`, `missing_m5_count`, `first_missing_utc`, `duplicate_count` et le statut du parseur V23. Une archive présentant réellement 8639 lignes ou une lacune ne doit PAS être réinterprétée comme 8640 barres : pas de remplissage, pas de tolérance ajoutée.

Si le fichier de rapport existe déjà, utiliser un **autre nom nouveau** ; ne jamais l'écraser. En cas d'échec SHA, stop immédiat avant tout diagnostic de prix. Aucun résultat du rapport ne prouve un edge B01/B02.
