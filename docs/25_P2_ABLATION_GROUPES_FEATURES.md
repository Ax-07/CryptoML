# CryptoML P2 — Audit causal de groupes de features H2 contre H6

Expérience CPU descriptive : 7 familles de features prédéfinies, ablation par réentraînement (full contre full moins une famille), comparaison appariée sur les mêmes observations et périodes de développement. Le modèle H6 figé n'est pas modifié.

## Protocole et garde-fous

- 30 features réparties exactement dans sept familles : rendements (6), tendance/momentum (3), volatilité (3), volume/flow (5), structure des bougies (5), calendrier (4), contexte BTC/ETH (4).
- Horizons **H2 et H6**, même coefficient de label 0,5, modèle LightGBM multiclasse CPU (150 estimators max), alpha figé à 1.0.
- 120 jours de train, 7 jours early stopping, 3 jours tuning non utilisés, 4 jours de vérification ; 14 jours OOS des folds et 90 jours finaux exclus.
- Pour garder les mêmes observations et dates pour les deux horizons, le jeu commun est construit avec tous les labels H1/H2/H3/H5/H6/H9/H12, mais aucun modèle H1/H3/H5/H9/H12 n'est entraîné ni évalué. Purge **H12** conservatrice pour les deux.
- Quatre actifs vus BTC, ETH, SOL, XRP ; ADA n'est même pas chargée.
- `full` vs ablation : mêmes périodes train/early/check, mêmes labels, mêmes hyperparamètres, seules les colonnes diffèrent. Référence de classe alpha recalculée uniquement depuis le train (labels propres à chaque horizon).
- Sortie par fold, horizon, famille et crypto : précision directionnelle, couverture, uplift, macro-F1, balanced accuracy, log loss et Brier.
- **Un bénéfice positif** signifie que garder le groupe est préférable à son retrait ; pour la log loss et le Brier, le bénéfice est la perte ablatée moins la perte full.
- Les importances gain/split LightGBM sont descriptives. Les ablations ne mesurent pas la causalité, notamment en raison des redondances entre features.
- Les 36 validations de développement ont déjà été étudiées à plusieurs reprises. Aucune indépendance statistique, sélection de modèle, rentabilité ni preuve de trading.

## Attention CPU

Avec 36 folds, 2 horizons et 8 variantes, il y a **576 entraînements CPU**. Prévoir un lancement interactif stable et de la mémoire ; le script refuse d'écraser un JSON existant. **La sauvegarde JSON finale n'a lieu qu'à la fin** : une interruption signifie relancer l'expérience depuis le début.

## Exécution PowerShell

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q
```

Après réussite des tests :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.feature_group_ablation `
  --data-root "E:\cryptoML\data" `
  --months 24 `
  --n-estimators 150 `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\feature_group_ablation_h2_h6_p2.json"
```

## Lecture des résultats

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\feature_group_ablation_h2_h6_p2.json" -Raw | ConvertFrom-Json

$tableau = foreach ($h in @("2", "6")) {
  foreach ($family in @("returns","trend_momentum","volatility","volume_flow","candle_shape","calendar","market_context")) {
    $x = $r.summary.group_ablation.$h.$family
    [PSCustomObject]@{
      Horizon = "H$h"
      Retrait = $family
      PrecisionBenefitPoints = [math]::Round($x.mean_benefit.directional_precision_benefit,3)
      UpliftBenefitPoints = [math]::Round($x.mean_benefit.uplift_points_benefit,3)
      CoverageBenefitPoints = [math]::Round($x.mean_benefit.coverage_benefit,3)
      LogLossBenefit = [math]::Round($x.mean_benefit.log_loss_benefit,5)
      FoldsBetterLogLoss = $x.full_better_log_loss_folds
    }
  }
}
$tableau | Format-Table -AutoSize
```

On ne supprimera pas automatiquement des variables parce qu'un groupe est défavorable sur la validation déjà observée. Les résultats orienteront une hypothèse de recherche, qui devra être confirmée sur des données prospectives nouvelles.
