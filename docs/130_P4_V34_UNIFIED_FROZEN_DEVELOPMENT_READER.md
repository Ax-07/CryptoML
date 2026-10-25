# P4 V34 — Lecteur unifié V21/V23/V33 et contrôle de préparation au développement

## Périmètre et interdictions
Branche de recherche isolée. V34 **ne lance aucun entraînement, backtest, PnL, export de prix, téléchargement ou ordre**. Aucune modification des historiques P0/V21/V23/V33, des journaux et des preuves V33. Le lecteur ne peut charger que BTCUSDT, ETHUSDT, SOLUSDT et XRPUSDT, pour les 21 mois **2024-10 à 2026-06 inclus**. Toute demande d'accès à 2026-07/08/09 ou ADA échoue **avant** d'accéder à un chemin source.

## Preuves utilisées
La référence SHA-256 de l'inventaire gelé est `36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71`; le plan V33 gelé a SHA `4cd4305d00f58095a0dfdc834a88ad2cd60ca14aef5fdb3484f0f9e3dbf56e6e`. Vérification hors réseau de l'inventaire SHA et de l'ensemble exact des sept preuves documentées (plan, V30, V32, résumé, journal, JSON/CSV de l'audit), et des 56 fichiers exacts des quatre partitions V33 (5 ZIP + 5 CHECKSUM + 4 artefacts).

V34 appelle l'audit live V33 qui vérifie les 84 partitions, puis confronte les 80 SHA des manifestes V21/V23 gelés et l'inventaire V33. Le registre n'est pas remplacé. V34 consulte ensuite chaque mois par son chemin canonique exact et revérifie les Parquet contre le manifeste avant lecture.

## Schémas et temps
Marché : `bar_open_at`, `bar_close_at` ; OHLC + volume spot/perp ; OHLC sans volume mark. Funding : `funding_at`, `funding_rate_fraction`, `funding_interval_hours`. Vérifier cadence M5, complétude mensuelle, synchronisme temporel, absence de NaN/inf, funding et continuité entre mois.

La barre ouverte le **2026-06-30T23:55:00Z** clôture exactement à **2026-07-01T00:00:00Z** : elle est vérifiée dans la partition native complète puis **exclue** du lecteur de développement. Quatre bougies sont exclues (une par actif). La plus tardive clôture éligible est donc `2026-06-30T23:55:00Z`. **Aucune lecture du holdout**.

## Exécution Windows : audit seulement

```powershell
Set-Location "E:\cryptoML-native-ingestion-v33"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : fetch" }
if (-not (Test-Path "E:\cryptoML-dev-reader-v34\backend")) {
    git worktree add "E:\cryptoML-dev-reader-v34" origin/research/p4-v34-unified-provenance-dev-reader
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree" }
}
Set-Location "E:\cryptoML-dev-reader-v34\backend"
$python="E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_unified_dev_reader_v34.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests ciblés" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }
$base="E:\cryptoML-walkforward-results"
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$j="$base\p4_v34_development_readiness_$stamp.json"
$c="$base\p4_v34_development_readiness_$stamp.csv"
& $python -m mlpipeline.funding_basis_unified_dev_reader_v34 `
    --freeze-json "$base\p4_v33_provenance_freeze_20261024_143113_6537528.json" `
    --freeze-sha256 "36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71" `
    --output-json $j `
    --output-csv $c
if ($LASTEXITCODE -ne 0) { throw "STOP : V34 pré-audit refusé" }
Get-Content $j -Raw
```

**Important :** les tests de régression sont synthétiques et doivent être exécutés localement. Aucun audit réel V34 n'a été réalisé avant réception du rapport de l'opérateur.

## Critère d'acceptation
84 mois-actifs vérifiés (1 V21 + 79 V23 + 4 V33), quatre séries continues de 21 mois, 4 bougies coupure exclues, aucune lecture juillet-septembre 2026, statuts inchangés, aucun prix exporté ni apprentissage. En cas de refus de gel SHA, schéma ou calendrier, **STOP sans publication**.
