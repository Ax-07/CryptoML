# CryptoML P2 — 120 / 240 / 365 jours d'entraînement sur dates communes

## Hypothèse

Augmenter la durée d'entraînement peut apporter plus d'exemples, mais aussi davantage de régimes anciens non pertinents. **48 mois d'archives ne signifient pas 48 mois appris par chaque modèle.** Avant de collecter davantage de données, comparer les trois durées sur les 24 mois P0 déjà disponibles.

## Comparaison contrôlée

- Modèle **LightGBM multiclasse CPU**, hyperparamètres identiques, 30 features inchangées, labels H6 (30 minutes) et seuil de volatilité 0,5 inchangés.
- Durées fixes **120, 240 et 365 jours** ; chaque modèle termine son entraînement à la **même date de début de validation**, mais commence plus ou moins tôt.
- Les folds communs sont définis d'après les **365 jours**, plus 14 jours de validation, 14 jours de test réservé, pas de chevauchement de tests (step 14 jours), et 90 jours finaux réservés.
- La validation identique comporte **7 jours early stopping**, 3 jours intermédiaires non utilisés, puis **4 jours de contrôle**. À chaque frontière, les labels dont la maturation dépasse la frontière sont purgés.
- **Alpha 1.0 fixé**, priors de classes calculés uniquement sur le train propre à chaque durée. Aucune sélection sur les contrôles.
- ADA exclue de l'entraînement et de toute métrique, OOS des folds et 90 jours finaux non évalués.
- Le test final a déjà été consulté auparavant et les validations de développement également : **ces résultats ne sont pas un nouveau test indépendant**.

Le JSON conserve chaque fold, périodes, tailles d'échantillons, iterations de LightGBM, scores probabilité brute (log loss et Brier) et décision alpha 1.0 (precision UP/DOWN, couverture, référence aléatoire, uplift, F1, balanced accuracy), séparations par crypto, moyennes et comparaisons appariées 240 vs 120 et 365 vs 120.

**Ne pas comparer directement les moyennes de cette expérience à celles des anciens 36 folds**, car seules les dates où 365 jours sont disponibles sont retenues. Comparer les trois durées *entre elles sur leurs folds communs*.

## Exécution (PowerShell)

Depuis le worktree **isolé** `E:\cryptoML-walkforward-test`, après `git fetch origin` et `git switch --detach origin/feat/cpu-purged-walkforward-folds` :

```powershell
cd E:\cryptoML-walkforward-test\backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.history_window_diagnostics `
  --data-root "E:\cryptoML\data" `
  --months 24 `
  --horizon 6 `
  --n-estimators 150 `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\train_windows_120_240_365_p2.json"
```

Le rapport refuse l'écrasement, les archives P0 restent lues uniquement et les modèles CPU déjà exportés ne sont pas modifiés.

Afficher :

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\train_windows_120_240_365_p2.json" -Raw | ConvertFrom-Json
$resultats = foreach ($jours in @(120,240,365)) {
  $s = $r.summary."$jours"
  [PSCustomObject]@{
    Jours = $jours
    Folds = $s.folds
    PrecisionPct = [math]::Round(100 * $s.directional_precision_mean,2)
    CouverturePct = [math]::Round(100 * $s.coverage_mean,2)
    UpliftPoints = [math]::Round($s.directional_uplift_points_mean,2)
    MacroF1 = [math]::Round($s.macro_f1_fixed_alpha_mean,4)
    LogLoss = [math]::Round($s.raw_log_loss_mean,4)
    Brier = [math]::Round($s.raw_multiclass_brier_mean,4)
  }
}
$resultats | Format-Table -AutoSize
$r.paired_comparisons | ConvertTo-Json -Depth 4
```

Une hausse de précision avec une baisse de couverture n'est pas automatiquement un gain. Analyser F1, log loss, uplift et stabilité des folds avant toute nouvelle expérience. Les horizons H3/H5/H6 pourront ensuite être comparés séparément, sans changer simultanément les deux dimensions.
