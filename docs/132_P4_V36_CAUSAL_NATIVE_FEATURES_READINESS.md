# P4 V36 — Contrat de features natives M5 causales, sans entraînement

V35 a validé 33 fenêtres walk-forward mais a bloqué le pipeline ML historique pour manque de volume_quote, trade_count et taker_buy_volume_base dans les sources V34, ainsi que l'absence de preuve de disponibilité temporelle du funding. V36 ne fabrique **aucune** de ces colonnes et n'exécute pas le modèle hérité.

## Contrat figé

V36 définit **19 features** dans une **nouvelle version de schéma** : rendements M5 et sur 12 M5 pour spot/perp/mark, volatilité glissante sur 48 rendements, écart à EMA26, amplitude OHLC relative de chaque marché, basis perp/mark et perp/spot, volumes relatifs 48 bougies pour spot et perp. Tout calcul porte sur des bougies déjà clôturées ; aucune interpolation, remplissage de valeurs manquantes ou volume fictif mark.

Les tests utilisent une série synthétique, la tronquent à t puis modifient radicalement son futur : les features calculées avant t ne doivent pas changer. Les trous M5, volumes négatifs, prix non finis, schémas incorrects et dates dans les périodes réservées déclenchent STOP.

**Important :** pour l'instant les calculs de disponibilité sont effectués par mois, avec un nouvel échauffement en début de partition ; cela n'est **pas** une spécification finale des features de production à cheval sur les mois. Les compteurs V36 servent de diagnostic, pas de nombre d'exemples utilisables pour l'entraînement.

## Barrières temporelles

- Source gelée V33 (SHA-256 `36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71`) et audit V35 SHA-256 doivent correspondre aux fichiers locaux. Pas de téléchargement.
- V36 lit uniquement **octobre 2024 à mai 2026** ; n'appelle pas le lecteur V34 pour juin 2026 et refuse toute entrée de juin ou au-delà.
- La bougie de mai ouverte à 23:55 le 31 mai et clôturant exactement le 1er juin 2026 est vérifiée mais exclue avant calcul.
- Les 33 fenêtres fixées en V35 restent des bornes temporelles théoriques. Aucun label, entraînement, score, backtest, ordre ou calcul PnL.

## Funding

Le champ événementiel `funding_at` n'est pas une preuve de publication antérieure. **Toutes les features funding sont exclues** ; une preuve indépendante de l'instant de disponibilité devra être approuvée avant toute nouvelle version.

## Exécution PowerShell : tests et métadonnées seulement

```powershell
Set-Location "E:\cryptoML-walkforward-preflight-v35"
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "STOP : git fetch" }
if (-not (Test-Path "E:\cryptoML-causal-features-v36\backend")) {
    git worktree add "E:\cryptoML-causal-features-v36" origin/research/p4-v36-causal-native-feature-contract
    if ($LASTEXITCODE -ne 0) { throw "STOP : worktree V36" }
}
Set-Location "E:\cryptoML-causal-features-v36\backend"
$python = "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe"
& $python -m pytest tests/test_funding_basis_causal_native_features_v36.py -q
if ($LASTEXITCODE -ne 0) { throw "STOP : tests V36" }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "STOP : suite globale" }
$base="E:\cryptoML-walkforward-results"
$freeze="$base\p4_v33_provenance_freeze_20261024_143113_6537528.json"
$v35=Get-ChildItem $base -Filter "p4_v35_walkforward_methodology_*.json" -File
if (@($v35).Count -ne 1) { throw "STOP : identifier un unique rapport V35" }
$v35Path=$v35[0].FullName
$v35Sha=(Get-FileHash $v35Path -Algorithm SHA256).Hash.ToLowerInvariant()
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fffffff"
$j="$base\p4_v36_native_causality_$stamp.json"
$c="$base\p4_v36_native_causality_$stamp.csv"
& $python -m mlpipeline.funding_basis_causal_native_features_v36 `
    --freeze-json $freeze `
    --freeze-sha256 "36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71" `
    --v35-json $v35Path `
    --v35-sha256 $v35Sha `
    --output-json $j --output-csv $c
if ($LASTEXITCODE -ne 0) { throw "STOP : audit V36" }
Get-Content $j -Raw
```

## Critère de réussite
Statut **`CAUSAL_NATIVE_FEATURE_DIAGNOSTICS_PASS_FUNDING_AND_ML_BLOCKED`** ; 19 features, 80 partitions source diagnostiquées (20 mois × 4 actifs), **175103 M5 admissibles avant le 1er juin par actif**, quatre barres frontière exclues. Le rapport ne stocke aucun prix, feature par ligne ou étiquette. Aucun protocole d'entraînement approuvé.
