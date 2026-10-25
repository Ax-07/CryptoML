# P4 V38 — audit strict des créneaux théoriques par fold (métadonnées uniquement)

V37 a validé 175055 lignes de features complètes par actif, soit 3648 récupérées au total grâce à la continuité inter-mois, avec les 19 features natives. V38 ne crée **aucune** cible supervisée et ne lit **aucun fichier de bougies**. Il lit seulement deux rapports JSON existants, identifiés par SHA-256 : V35 (calendrier et 33 folds) et V37 (19 features, source gelée V33 et réconciliation V36).

Le décompte est celui des **créneaux M5 théoriquement admissibles**, pas celui des labels réellement calculables, des prédictions, des candidats de trading ou des observations uniques. Les 33 fenêtres d'entraînement roulantes se chevauchent : sommer leurs comptes ne revient pas à mesurer un dataset unique.

La première ligne de features complètes est close le 2024-10-01 04:05 UTC (48 bougies initiales incomplètes). La maturité théorique d'une étiquette à horizon 6 M5 doit être **strictement antérieure** à la fin exclusive de chaque sous-fenêtre. L'embargo V35 de 6 bougies s'applique après l'ouverture des fenêtres de validation et test. Juin 2026 reste dans la réserve interne aveugle ; juillet–septembre 2026 dans le holdout externe inaccessible.

Le financement reste exclu et la définition économique / supervisée de la cible demeure **bloquée**, sans entraînement ni PnL.

## Commandes PowerShell

```powershell
Set-Location "E:\cryptoML-cross-month-v37"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : git fetch" }
if (-not (Test-Path "E:\cryptoML-fold-eligibility-v38\backend")) {
    git worktree add "E:\cryptoML-fold-eligibility-v38" origin/research/p4-v38-sealed-supervised-eligibility-audit
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree" }
}
Set-Location "E:\cryptoML-fold-eligibility-v38\backend"
$python="E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_fold_eligibility_v38.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V38" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite complète" }
$base="E:\cryptoML-walkforward-results"
$v35=@(Get-ChildItem $base -File -Filter "p4_v35_walkforward_methodology_*.json")
$v37=@(Get-ChildItem $base -File -Filter "p4_v37_cross_month_causal_*.json")
if ($v35.Count -ne 1 -or $v37.Count -ne 1) { throw "STOP : rapports V35/V37 ambigus" }
$v35sha=(Get-FileHash $v35[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$v37sha=(Get-FileHash $v37[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$j="$base\p4_v38_fold_eligibility_$stamp.json"
$c="$base\p4_v38_fold_eligibility_$stamp.csv"
& $python -m mlpipeline.funding_basis_fold_eligibility_v38 `
  --v35-json $v35[0].FullName --v35-sha256 $v35sha `
  --v37-json $v37[0].FullName --v37-sha256 $v37sha `
  --output-json $j --output-csv $c
if ($LASTEXITCODE -ne 0) { throw "STOP : audit V38" }
Get-Content $j -Raw
```

Résultat cible : `THEORETICAL_FOLD_ELIGIBILITY_PASS_LABELS_AND_ML_BLOCKED`. Les comptes agrégés attendus sont 4 560 804 créneaux train, 530 640 validation et 530 640 test **sur l'ensemble des folds et des actifs, donc avec répétition entre folds**. Ces sommes ne sont pas des exemples étiquetés. Toute divergence ou changement de SHA impose STOP et audit.
