# P4 V40 — Approbation limitée de la mathématique cible, tests synthétiques seulement

## Décision

La proposition V39 a reçu accord pour **la définition de la cible** : marché perp, clôture native M5 à t, horizon exactement H=6 clôtures (30 minutes), écart de prix futur `perp_close(t+6)/perp_close(t)-1`, seuil `0.5 * perp_volatility_48(t) * sqrt(6)` et trois classes DOWN / NEUTRAL / UP. Aux seuils exacts, la classe est NEUTRAL. À chaque instant t, la volatilité repose uniquement sur les 48 rendements logarithmiques passés et la bougie clôturée t. Le futur sert uniquement à définir le label, **jamais une feature**.

Le changement de statut n'approuve **ni** production d'étiquettes historiques **ni** lancement d'un entraîneur. La fonction de calcul V40 est pure et doit être utilisée **exclusivement avec des fixtures synthétiques en mémoire**, indépendamment du lecteur V34. L'interface en ligne de commande ne charge aucun prix, aucun funding, aucun historique, n'appelle jamais cette fonction de labels et n'écrit qu'un JSON de conformité.

## Garde-fous

- Document normatif V39 épinglé par Git blob SHA-1 `48445865b9ae7c19f25be0927a375e5188374d32` et inventorié en SHA-256 ; rapport V38 contrôlé par SHA-256 fourni explicitement par l'opérateur. Lignée V38 : gel V33 `36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71`, V35 `ad478769e6b3d58f3df7143fb8d5f917937a3f10eba090f5747700a2cd8e2abe`, V37 `33308282142445e912faa1a8ac215e38e0e9764d5ff97f7e64859410f1235d08`.
- Schéma de features V36 `p4_native_closed_m5_features_v36_v1`, sans financement faute de publication/as-of attestée.
- Dates UTC, fenêtres M5 strictement approuvées ; rejet des trous, bougies partielles, nan/inf, OHLC invalides et toute période de juin 2026 et au-delà.
- Purge stricte `label_mature_at < window_end_exclusive` ; embargo de 6 M5 aux débuts validation/test ; la frontière de fin n'est jamais incluse.
- Aucun ajustement de paramètres avec juin 2026 ou juillet-septembre 2026. Pas de PnL/frais/trades.

## Installation Windows et audit V40

```powershell
Set-Location "E:\cryptoML-fold-eligibility-v38"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : fetch" }
if (-not (Test-Path "E:\cryptoML-target-spec-v40\backend")) {
    git worktree add "E:\cryptoML-target-spec-v40" origin/research/p4-v40-approved-target-synthetic-only
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree" }
}
Set-Location "E:\cryptoML-target-spec-v40\backend"
$python="E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_target_contract_v40.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V40" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }

$base="E:\cryptoML-walkforward-results"
$v38=@(Get-ChildItem $base -Filter "p4_v38_fold_eligibility_*.json" -File)
if ($v38.Count -ne 1) { throw "STOP : un unique rapport V38 obligatoire" }
$sha=(Get-FileHash $v38[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$out="$base\p4_v40_approved_synthetic_target_$stamp.json"

& $python -m mlpipeline.funding_basis_target_contract_v40 `
    --v39-document "..\docs\135_P4_V39_PROPOSED_CAUSAL_TARGET_CONTRACT.md" `
    --v38-json $v38[0].FullName `
    --v38-sha256 $sha `
    --output-json $out `
    --approve-synthetic-only
if ($LASTEXITCODE -ne 0) { throw "STOP : approbation V40" }
Get-Content $out -Raw
```

## Condition d'acceptation

Statut attendu : `TARGET_CONTRACT_APPROVED_SYNTHETIC_ONLY_HISTORICAL_LABELS_BLOCKED`.

Le rapport indique explicitement `historical_label_creation=NOT_AUTHORIZED`, `training=NOT_AUTHORIZED`, absence de lecture des données de marché, absence d'export de labels et absence de PnL/ordres. Aucun compte de labels historiques n'est produit.
