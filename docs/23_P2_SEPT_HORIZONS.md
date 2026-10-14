# CryptoML P2 — Sept horizons M5 sur dates communes

## Question

Après le premier comparatif H3/H5/H6 (H3 : précision 32,61 %, H6 : 31,97 %, 22/36 folds favorables à H3), tester **H1, H2, H3, H5, H6, H9 et H12** sans changer l'architecture, les features, les hyperparamètres, la taille de train ou la définition des labels.

## Protocole strict

- H1=5, H2=10, H3=15, H5=25, H6=30, H9=45 et H12=60 minutes.
- 4 cryptos déjà vues : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT. **ADAUSDT n'est ni chargée ni évaluée**. BTC et ETH restent le contexte causal habituel.
- Données : 24 partitions mensuelles P0 M5 existantes et vérification optionnelle des SHA256, jamais d'écriture dans P0.
- Features construites une fois par symbole ; labels `mlpipeline.labels.add_labels` sur bougies contiguës avec le même coefficient de seuil 0,5 et horizon variable.
- Intersection stricte des timestamps où les 7 labels et les 30 features sont valides. La date de maturité utilisée pour TOUTES les partitions est celle de **H12**, plus conservatrice que H1/H2/H3/H5/H6/H9.
- Folds identiques : 120 jours train, 7 jours early stop, 3 jours tampon de réglage **inutilisé**, 4 jours de check ; 14 jours OOS réservés non notés, pas 14 jours de chevauchement de tests, derniers 90 jours non évalués.
- CPU LightGBM multiclasse, 150 estimateurs maximum, alpha fixe 1,0, mêmes 30 features.
- Pour chaque horizon, prior de train propre aux labels de cet horizon, proportions DOWN/NEUTRAL/UP check, log loss et Brier brutes, skill de log loss par rapport aux probas du prior train, précision directionnelle, couverture, référence aléatoire et uplift.
- Comparaisons appariées **par fenêtre et par symbole**, référence H6. Ne pas sélectionner mécaniquement « le meilleur » horizon : plusieurs hypothèses ont été regardées successivement sur les validations de développement, **pas de confirmation indépendante**.
- Aucun nouveau modèle exporté, aucune exécution de trading.

## Commandes Windows PowerShell

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.expanded_horizon_diagnostics `
  --data-root "E:\cryptoML\data" `
  --months 24 `
  --n-estimators 150 `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\horizons_h1_h2_h3_h5_h6_h9_h12_p2.json"
```

N'exécuter l'expérience que si les tests passent. Le JSON refuse l'écrasement.

### Tableau de synthèse

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\horizons_h1_h2_h3_h5_h6_h9_h12_p2.json" -Raw | ConvertFrom-Json
$resultats = foreach ($h in @(1,2,3,5,6,9,12)) {
  $s = $r.summary."$h"
  [PSCustomObject]@{
    Horizon = "H$h"
    Minutes = $s.minutes_ahead
    Folds = $s.folds
    PrecisionPct = [math]::Round(100 * $s.directional_precision_mean, 2)
    CouverturePct = [math]::Round(100 * $s.directional_coverage_mean, 2)
    UpliftPoints = [math]::Round($s.directional_uplift_points_mean, 2)
    MacroF1 = [math]::Round($s.macro_f1_mean, 4)
    SkillLogLossPct = [math]::Round(100 * $s.log_loss_skill_vs_train_prior_mean, 2)
    NeutralPct = [math]::Round(100 * $s.class_distribution_mean.NEUTRAL, 2)
  }
}
$resultats | Format-Table -AutoSize
$r.paired_vs_h6 | ConvertTo-Json -Depth 4
```

La granularité par actif est disponible dans `$r.by_symbol_summary`. La granularité par fold dans `$r.folds`.

**Important :** ce nouveau calcul sur intersection H12 peut donner de petites variations par rapport à l'ancienne expérience H3/H5/H6 (intersection H6). Les comparaisons *internes à ce rapport* sont les seules parfaitement appariées.
