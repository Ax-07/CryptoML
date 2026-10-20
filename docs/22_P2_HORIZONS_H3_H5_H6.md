# CryptoML P2 — Horizon H3 / H5 / H6 (CPU)

Le diagnostic des profondeurs d'apprentissage (18 folds communs) a montré une amélioration légère de log loss avec 365 jours, mais aucune amélioration convaincante de la précision UP/DOWN. L'étape suivante isole **l'horizon** avec **120 jours d'entraînement** pour ne modifier qu'une seule dimension.

## Protocole
- Bougies M5, horizons H3=15 minutes, H5=25 minutes et H6=30 minutes ; coefficient de seuil de labels 0,5 multiplié par la volatilité passée et sqrt(horizon).
- Même LightGBM 3 classes CPU, mêmes 30 features, alpha fixe 1.0, 120 jours de train, 7 jours early stop, 3 jours buffer non utilisé, 4 jours vérification.
- Alignement strict sur l'intersection des **symboles + timestamps M5** où les trois horizons sont connus et les features finies. Partitions absolument identiques pour les trois horizons. Purge **conservatrice H6** pour tous, même H3/H5.
- ADA absente des évaluations et des apprentissages ; aucun fold OOS ni derniers 90 jours consultés.
- Les données historiques déjà consultées sont des **validations de développement, non indépendantes**.
- Les fréquences DOWN/NEUTRAL/UP évoluent avec H. Comparez notamment **l'avantage directionnel vs références de classes propres à chaque horizon** et la **réduction relative de log loss vs prédictions du prior appris**, plutôt que l'accuracy brute ou la log loss seule.
- Aucun remplacement du modèle figé, aucun ordre réel.

## Exécution PowerShell
```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.horizon_diagnostics `
  --data-root "E:\cryptoML\data" `
  --months 24 `
  --n-estimators 150 `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\horizons_h3_h5_h6_p2.json"
```

Ne lancer l'expérience que si les tests passent. Le JSON refuse d'écraser un rapport existant.

## Résumé PowerShell
```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\horizons_h3_h5_h6_p2.json" -Raw | ConvertFrom-Json
$resultats = foreach ($h in @(3,5,6)) {
  $s = $r.summary."$h"
  [PSCustomObject]@{
    Horizon = "H$h"
    Minutes = $s.minutes_ahead
    Folds = $s.folds
    PrecisionPct = [math]::Round(100*$s.directional_precision_mean,2)
    CouverturePct = [math]::Round(100*$s.directional_coverage_mean,2)
    AvantagePoints = [math]::Round($s.directional_uplift_points_mean,2)
    MacroF1 = [math]::Round($s.macro_f1_mean,4)
    SkillLogLossPct = [math]::Round(100*$s.log_loss_skill_vs_train_prior_mean,2)
    NeutralPct = [math]::Round(100*$s.class_distribution_mean.NEUTRAL,2)
  }
}
$resultats | Format-Table -AutoSize
```

Tous les résultats devront être interprétés en tenant compte du changement de définition du label, du caractère déjà consulté des vérifications de développement et de la corrélation des étiquettes M5 glissantes.
