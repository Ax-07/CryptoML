# P2 - Benchmark directionnel simple H2/H6

Objectif: comparer direction majoritaire du train, momentum et reversion sur 1/3/6 bougies M5, et LightGBM 30 features (150 estimateurs, seed 42). 36 folds 120d train, 7d early stop, 3d tuning sans labels pour la sélection, 4d check; purge H12 60 minutes.

Vues distinctes: all_check = choix binaire UP/DOWN sur chaque ligne, NEUTRAL compte comme erreur; oracle_true_movers = vrais mouvements connus apres coup, donc NON deployable; shared_ml_selected = memes lignes selectionnees par le filtre LightGBM a 40% de couverture cible tuning, les regles simples ne sont pas des strategies autonomes. Le ML choisit son signe via p_up contre p_down, ce qui differe de sa decision multiclasses classique. Aucune proba calibree n'est attribuee aux regles; pas de Brier/logloss pour celles-ci.

Les features et labels sont calcules avec les archives BTC/ETH/SOL/XRP en lecture seule et les labels sont construits APRES troncature a la limite de check de CHAQUE fold. Aucune evaluation OOS/final 90 jours ni ADA, aucune modification des modeles P3 ou des packs forward. Checks deja inspectes : resultats exclusivement exploratoires.

## Commandes PowerShell (checkout isole)
```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Smoke technique (1 fold, 2 entrainements):
```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.directional_rule_benchmark `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

Apres smoke vert: 36 folds et 72 entrainements CPU:
```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.directional_rule_benchmark `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" `
  --months 24 --n-estimators 150 --seed 42 --verify-hashes
```

## Bilan PowerShell
```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = foreach ($h in @("H2","H6")) {
  foreach ($view in @("all_check","oracle_true_movers","shared_ml_selected")) {
    foreach ($rule in $r.rules) {
      $m = $r.summary.$h.$view.$rule
      [PSCustomObject]@{
        Horizon = $h; Vue = $view; Regle = $rule
        Folds = $m.comparable_folds
        PrecisionPct = [math]::Round(100*$m.mean_precision,2)
        GainMajoritePts = [math]::Round($m.mean_gain_vs_train_majority_points,3)
        FoldsFavorables = $m.folds_beating_train_majority
      }
    }
  }
}
$rows | Format-Table -AutoSize
```

Les folds ne sont pas independants. Ne pas choisir automatiquement une regle gagnante, ne pas confondre couverture de tuning et couverture realisee en check.