# P4 V37 — continuité causale M5 inter-mois ; AUCUN entraînement

V36 a diagnostiqué 19 features natives sur 20 mois par actif, en réinitialisant le calcul à chaque mois : **175103 barres** avant la réserve de juin et **174143 lignes complètes** par actif, soit 960 échauffements (20 × 48). V37 enchaîne les données natives vérifiées avant de calculer ces mêmes features, sans modifier le schéma ou les sources.

## Contrat et invariants

- Figer et revalider par SHA-256 le gel V33, le rapport V35, et le rapport V36. Les trois liens de provenance doivent correspondre.
- Accéder exclusivement à BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT du **2024-10 au 2026-05**, via le lecteur V34 et ses audits de sources. Ne **jamais** demander le mois 2026-06 ni le holdout 2026-07 à 2026-09.
- Dernière bougie M5 du 31 mai 2026 clôture le 1er juin et est **exclue** avant les features ; ne lire aucune bougie de juin.
- Refuser tous doublons/gaps M5 inter-mois, vérifier que chaque mois a une grille native et que les OHLC/volumes passent le contrat V36.
- Concaténer les vingt mois **en mémoire**, puis calculer les 19 features natives sur le flux causal unique. Les indicateurs EMA et glissants continuent le mois suivant ; pas de lookahead, pas de remplissage ni interpolation. Vérifier l'invariance de deux préfixes à des frontières mensuelles.
- La comparaison de disponibilité attendue est : `175103 - 48 = 175055` lignes complètes par actif ; **912 récupérées** par rapport au V36 (19 frontières × 48), donc **3648 de plus** sur quatre actifs. Il s'agit de lignes de *features* complètes, **pas d'étiquettes ni de candidats ML**.
- Chaque mois présente 48 premiers indicateurs incomplets sans carryover (V36), dont les 19 mois suivants devraient être rendus complets par la seule continuité historique. Funding hors features tant que `funding_published_at` n'est pas attesté.
- Écrire uniquement un JSON de comptes/état et un CSV mensuel de métadonnées, fichiers **nouveaux et hors dossiers protégés**. Aucun export de prix ou features par ligne, aucun entraînement, label, score, PnL ou ordre.

## PowerShell Windows

```powershell
Set-Location "E:\cryptoML-causal-features-v36"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : git fetch" }
if (-not (Test-Path "E:\cryptoML-cross-month-v37\backend")) {
    git worktree add "E:\cryptoML-cross-month-v37" origin/research/p4-v37-cross-month-causal-feature-continuity
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree V37" }
}
Set-Location "E:\cryptoML-cross-month-v37\backend"
$python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_cross_month_features_v37.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V37" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }
$base = "E:\cryptoML-walkforward-results"
$v35 = @(Get-ChildItem $base -Filter "p4_v35_walkforward_methodology_*.json" -File)
$v36 = @(Get-ChildItem $base -Filter "p4_v36_native_causality_*.json" -File)
if ($v35.Count -ne 1 -or $v36.Count -ne 1) {
    throw "STOP : rapports V35/V36 non uniques"
}
$v35sha = (Get-FileHash $v35[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$v36sha = (Get-FileHash $v36[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$json = "$base\p4_v37_cross_month_causal_$stamp.json"
$csv = "$base\p4_v37_cross_month_causal_$stamp.csv"
& $python -m mlpipeline.funding_basis_cross_month_features_v37 `
    --freeze-json "$base\p4_v33_provenance_freeze_20261024_143113_6537528.json" `
    --freeze-sha256 "36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71" `
    --v35-json $v35[0].FullName --v35-sha256 $v35sha `
    --v36-json $v36[0].FullName --v36-sha256 $v36sha `
    --output-json $json --output-csv $csv
if ($LASTEXITCODE -ne 0) { throw "STOP : audit V37" }
Get-Content $json -Raw
```

Statut espéré `CROSS_MONTH_CAUSAL_CONTINUITY_PASS_ML_AND_FUNDING_BLOCKED` ; les tests et le diagnostic réel restent à réaliser par l'opérateur. Le protocole d'entraînement futur, son univers de validation et la gestion des labels devront faire l'objet d'une autorisation séparée.
