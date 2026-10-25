# P4 V35 — Pré-audit du calendrier walk-forward, strictement sans ML

## Fondations
V34 a contrôlé **84/84** partitions pour BTCUSDT ETHUSDT SOLUSDT XRPUSDT entre 2024-10 et 2026-06, soit 183743 clôtures M5 de développement par actif et 4 bougies frontières exclues. Le SHA du gel V33 est `36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71`.

V35 appelle le lecteur V34, revalide hors réseau cette provenance gelée et l'intégrité des 84 partitions ; aucun appel API Binance, entraînement, préparation d'étiquettes, génération de features, calcul PnL ou export de prix.

## Géométrie fixée avant apprentissage
- Univers : 4 actifs ; observations de développement uniquement 2024-10-01T00:00 UTC au 2026-07-01T00:00 UTC **exclusif**.
- **Réserve de développement aveugle** : juin 2026, ne fait partie d'aucune fenêtre d'entraînement, validation ou test V35.
- **Véritable holdout externe** : juillet–septembre 2026, les chemins correspondants ne sont jamais ouverts.
- Walk-forward à entraînement roulant 120 jours, validation 14 jours, test 14 jours, pas de 14 jours ; aucune fenêtre test chevauchante et aucun test après le 1er juin 2026.
- Horizon d'étiquette théorique : 6 bougies M5 = 30 minutes, cohérent avec `mlpipeline.labels.add_labels`. Entraînement et validation exigent `label_mature_at < fin_exclusive`, de même pour test. Il s'agit de bornes géométriques **théoriques**, pas d'étiquettes créées.
- En complément du purge, embargo de 6 M5 (30 minutes) au début des fenêtres de validation et test. Compte théorique donné à titre indicatif, **pas le nombre réel d'exemples éligibles**.

## Blocages explicites du ML
Le `FEATURE_SCHEMA_VERSION=features_v1` historique nécessite volume_quote, trade_count et taker_buy_volume_base. Les Parquet V34 ne possèdent pas ces champs ; la disponibilité des champs OHLC, volume_base et mark/funding n'autorise pas à fabriquer les autres. Il faut créer un adaptateur V36 traçable ou une nouvelle spécification de features déclarée et approuvée, sans réutiliser silencieusement le modèle existant.

Les archives funding contiennent la date d'événement `funding_at` mais n'attestent pas une colonne de **publication/disponibilité au modèle**. L'utilisation funding dans une feature à date t est donc **bloquée** avant clarification du timing, indépendamment de son intégrité cryptographique.

La procédure historique `mlpipeline.walkforward_train` utilise un mécanisme de holdout d'actif (exemple ADAUSDT) distinct et ses réglages par défaut ne doivent pas être implicitement transposés à l'univers V34. V35 ne déclenche jamais cette procédure.

## Commandes Windows

```powershell
Set-Location "E:\cryptoML-dev-reader-v34"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : fetch" }
if (-not (Test-Path "E:\cryptoML-walkforward-preflight-v35\backend")) {
    git worktree add "E:\cryptoML-walkforward-preflight-v35" origin/research/p4-v35-walkforward-methodology-preflight
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree" }
}
Set-Location "E:\cryptoML-walkforward-preflight-v35\backend"
$python="E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_walkforward_preflight_v35.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V35" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }
$base="E:\cryptoML-walkforward-results"
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$json="$base\p4_v35_walkforward_methodology_$stamp.json"
$csv="$base\p4_v35_walkforward_methodology_$stamp.csv"
& $python -m mlpipeline.funding_basis_walkforward_preflight_v35 `
    --freeze-json "$base\p4_v33_provenance_freeze_20261024_143113_6537528.json" `
    --freeze-sha256 "36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71" `
    --output-json $json --output-csv $csv
if ($LASTEXITCODE -ne 0) { throw "STOP : pré-audit V35" }
Get-Content $json -Raw
```

**Statut attendu :** `CALENDAR_PREFLIGHT_PASS_ML_PIPELINE_BLOCKED`, car V35 établit la possibilité du découpage et signale honnêtement les incompatibilités des features et l'incertitude de disponibilité funding. Ne lancer aucun entraînement.
