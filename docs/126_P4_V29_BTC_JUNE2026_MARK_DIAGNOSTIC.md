# V29 — Diagnostic ciblé mark-price BTCUSDT juin 2026

**État préenregistré :** V24 80/84 vérifiées, 4 manquantes, zéro FAILED_INTEGRITY. Une deuxième tentative V27 a imprimé `spot verified` puis `perp verified` avant de s'arrêter sur `Incorrect M5 count/shape`. Le troisième type traité par V23 est `mark` : le défaut est donc localisé dans le décodage markPriceKlines, *pas dans spot*. La validation spot V28 à 8640 lignes ne résout pas ce défaut.

**Une seule archive publique lue :** `futures/um/monthly/markPriceKlines/BTCUSDT/5m/BTCUSDT-5m-2026-06.zip` et son `.CHECKSUM`, avec vérification du SHA256, contrôle de l'unique CSV ZIP et des limites de décompression. Rapport JSON nouveau, aucun fichier ZIP ni parquet conservé, aucune écriture P0/V21/V23, aucune modification du parseur V23. Les 8640 timestamps M5 juin 2026, colonnes, début/fin, doublons, missing et `v23_mark_parser_status` sont rapportés. Ne pas relancer V27 avant examen.

## PowerShell

```powershell
Set-Location "E:\cryptoML-controlled-batches-v27"
git fetch origin
git worktree add "E:\cryptoML-mark-diagnostic-v29" origin/research/btc-june2026-mark-archive-diagnostic-v29
Set-Location "E:\cryptoML-mark-diagnostic-v29\backend"
$python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_mark_diagnostic_v29.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V29" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complete" }
& $python -m mlpipeline.funding_basis_mark_diagnostic_v29 `
  --permit-downloads `
  --output-json "E:\cryptoML-walkforward-results\p4_v29_btc_2026_06_mark_diagnostic_v1.json"
if ($LASTEXITCODE -ne 0) { throw "STOP : diagnostic V29" }
Get-Content "E:\cryptoML-walkforward-results\p4_v29_btc_2026_06_mark_diagnostic_v1.json" -Raw
```

En cas de fichier déjà existant, employer un nom de sortie nouveau ; ne jamais l'écraser. Si l'archive présente moins de 8640 barres, on ne comble pas les lacunes sans modèle de provenance et revue explicites ; les résultats B01/B02 ne doivent pas être lancés sur des données non validées.
