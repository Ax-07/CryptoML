# V32 — Pré-vérification des sources mark ETH/SOL/XRP, juin 2026

Constat V31 : chaque archive mark mensuelle a un SHA-256 Binance valide mais contient 8352/8640 barres et 288 timestamps manquants à partir du 29 juin 2026. V30 confirme pour BTC que le quotidien du 29 juin contient les 288 barres manquantes ; statut uniquement candidat, aucune publication.

V32 étend le contrôle en **lecture seule** à ETHUSDT, SOLUSDT, XRPUSDT. Pour chacun : vérifier SHA-256 de ZIP mensuel et ZIP quotidien, exiger le SHA mensuel constaté lors de V31, vérifier que les 288 timestamps absents sont exactement le 29 juin, 288 timestamps du quotidien sans overlap, grille complète 8640 M5, vérifier 29 autres jours par le parseur V23. Ne pas imputer les barres, ne pas publier de Parquet ou modifier P0, V21, V23, V24, V27. Un résultat éligible est seulement **une candidature** pour une ingestion versionnée ultérieure, avec provenance des deux ZIP et un audit indépendant adapté.

## PowerShell

```powershell
Set-Location "E:\cryptoML-controlled-batches-v27"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : git fetch" }
if (-not (Test-Path "E:\cryptoML-mark-daily-v32\backend")) {
    git worktree add "E:\cryptoML-mark-daily-v32" origin/research/june2026-daily-mark-reconciliation-v32
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree" }
}
Set-Location "E:\cryptoML-mark-daily-v32\backend"
$python="E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_remaining_daily_mark_reconciliation_v32.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests ciblés" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests globaux" }
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$report="E:\cryptoML-walkforward-results\p4_v32_june2026_mark_daily_$stamp.json"
& $python -m mlpipeline.funding_basis_remaining_daily_mark_reconciliation_v32 `
    --permit-downloads --output-json $report
if ($LASTEXITCODE -ne 0) { throw "STOP : V32" }
Get-Content $report -Raw
```

Si l'archive quotidienne manque ou son checksum échoue, stop ; aucune réparation BUSD ni promotion silencieuse.
