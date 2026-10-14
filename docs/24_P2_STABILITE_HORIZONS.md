# CryptoML P2 — Vérification exploratoire de H2 par actif et par période

La comparaison des sept horizons a montré H2 à 32,93 % de précision, 64,32 % de couverture, +4,52 points d'avantage directionnel et 26/36 folds de skill log loss meilleur que H6. Mais ces folds ne sont pas indépendants, et H2 ne gagne la précision directionnelle que sur 22/36 folds.

Avant tout changement de feature ou de modèle, `mlpipeline.horizon_stability_report` **lit uniquement le JSON des sept horizons existant** et compare H1/H2/H3 à H6 au niveau :
- des quatre cryptos vues, séparément ;
- des deux moitiés chronologiques de folds ;
- des six groupes chronologiques de six folds ;
- des prédictions UP et DOWN pour H1/H2/H3/H6 ;
- des moyennes de précision/uplift/couverture **appariées**, et de la précision micro-pooled (distincte de la moyenne des folds).

Aucun nouvel entraînement, aucune lecture P0, aucune consultation d'ADA ou des réserves OOS/finale. **Il s'agit d'un diagnostic post hoc**, et non d'une validation indépendante du choix de H2. Ne pas interpréter les folds comme des essais indépendants ni la précision directionnelle comme un taux de profit.

## Exécution PowerShell, worktree isolé

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.horizon_stability_report `
  --input "E:\cryptoML-walkforward-results\horizons_h1_h2_h3_h5_h6_h9_h12_p2.json" `
  --output "E:\cryptoML-walkforward-results\horizon_stability_h2_p2.json"
```

## Sorties de synthèse

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\horizon_stability_h2_p2.json" -Raw | ConvertFrom-Json

$actifs = foreach ($sym in @("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")) {
    $x = $r.by_asset.$sym.comparisons.H2_vs_H6
    [PSCustomObject]@{
        Actif = $sym
        Folds = $x.folds_with_both_precisions_defined
        GainsPrecision = $x.wins_precision
        GainPrecisionPoints = [math]::Round($x.mean_precision_delta_points, 3)
        GainUpliftPoints = [math]::Round($x.mean_uplift_delta_points, 3)
        EcartCouverturePoints = [math]::Round($x.mean_coverage_delta_points, 3)
    }
}
$actifs | Format-Table -AutoSize

$periodes = foreach ($key in @("earlier_half","later_half",
  "chronological_block_01","chronological_block_02",
  "chronological_block_03","chronological_block_04",
  "chronological_block_05","chronological_block_06")) {
    $p = $r.by_time.$key
    $x = $p.comparisons.H2_vs_H6
    [PSCustomObject]@{
        Periode = $key
        Folds = $x.folds
        GainsPrecision = $x.wins_precision
        GainPrecisionPoints = [math]::Round($x.mean_precision_delta_points, 3)
        GainUpliftPoints = [math]::Round($x.mean_uplift_delta_points, 3)
    }
}
$periodes | Format-Table -AutoSize
```

La lecture des quatre cryptos et de la stabilité temporelle doit précéder un audit des features. La base H6 figée reste inchangée.
