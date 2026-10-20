# CryptoML P2 — Diagnostic des régimes de marché (H2 / H6)

## Question de recherche

Les effets des familles \`calendar\` et \`volatility\` étaient contrastés selon les actifs. Vérifier de manière descriptive si la précision et le signal directionnel de H2 (10 minutes) varient avec l'heure UTC, les blocs de six heures, semaine/week-end et la volatilité **connue au moment de la prédiction**, face à H6 (30 minutes).

## Sécurité et méthodes

- Deux LightGBM multiclasse CPU **complets** par fold (H2 et H6), 30 features, alpha 1.0, hyperparamètres fixés, 120 jours de train, 7 jours early stop, tampon 3 jours inutilisé, 4 jours de validation descriptive.
- Observations M5 exactement communes, avec **purge conservatrice H12 (60 minutes)**. Pas d'OOS fold, pas des 90 jours finaux, pas d'ADA chargée, pas d'ordres.
- Aucun changement de modèle/feature/label ; aucun réglage de stratégie sur les sous-groupes.
- Groupes prédéclarés : 24 heures UTC, quatre blocs \`00-05/06-11/12-17/18-23\`, semaine vs week-end, et tertiles de volatilité \`volatility_48\`.
- **Seuils des tertiles calculés exclusivement sur les 120 jours train**, et séparément pour BTC, ETH, SOL, XRP, fold par fold. La volatilité de la bougie close actuelle est une feature causale. Jamais de quantile ajusté sur les checks futurs.
- Pour chaque cellule : observations, nombre de décisions UP/DOWN, couverture, précision et erreurs NEUTRAL, référence aléatoire à partir des proportions de classes **de la cellule et du fold**, uplift conditionnel aux fréquences observées, résultats par symbole.
- Pour H2/H6, comparaison des précisions, uplift et couverture sur les **mêmes cellules et folds**. Les folds avec moins de 20 prédictions directionnelles sur au moins un horizon sont exclus des comparaisons fold-par-fold, mais les observations et effectifs restent visibles.
- Les cellules rares, la multiplicité des comparaisons et les validations déjà consultées interdisent toute affirmation de robustesse statistique indépendante ou de rentabilité. Les horaires « favorables » ne doivent pas être transformés en règles de trading.

## Windows PowerShell

Dans le worktree isolé uniquement :

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q
\`\`\`

Après des tests verts :

\`\`\`powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.market_regime_diagnostics \`
  --data-root "E:\cryptoML\data" \`
  --months 24 \`
  --n-estimators 150 \`
  --verify-hashes \`
  --output "E:\cryptoML-walkforward-results\market_regimes_h2_h6_p2.json"
\`\`\`

Seulement 72 fits CPU (36 folds × 2 modèles). Le rapport JSON refuse d'écraser un fichier existant ; il est enregistré à la fin.

## Synthèse à copier-coller

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\market_regimes_h2_h6_p2.json" -Raw | ConvertFrom-Json
$regimes = foreach ($axe in @("utc_block_6h","day_type","volatility_48_train_tercile")) {
    foreach ($prop in $r.summary.$axe.PSObject.Properties) {
        $x = $prop.Value.ALL
        [PSCustomObject]@{
            Axe = $axe
            Groupe = $prop.Name
            PredictionsH2 = $x.H2.selected
            PrecisionH2Pct = [math]::Round(100 * $x.H2.pooled_precision, 2)
            UpliftH2Points = [math]::Round($x.H2.pooled_uplift_points, 2)
            CouvertureH2Pct = [math]::Round(100 * $x.H2.pooled_coverage, 2)
            PrecisionH6Pct = [math]::Round(100 * $x.H6.pooled_precision, 2)
            DeltaUpliftMoyen = [math]::Round($x.paired.H2_minus_H6_uplift_points_mean, 2)
            FoldsApparies = $x.paired.folds_with_both_min_20_selected
        }
    }
}
$regimes | Format-Table -AutoSize
\`\`\`

Puis l'examen de la variation par crypto :

\`\`\`powershell
$actifs = foreach ($sym in @("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")) {
    foreach ($vol in @("low","medium","high")) {
        $x = $r.summary.volatility_48_train_tercile.$vol.$sym
        [PSCustomObject]@{
            Crypto = $sym
            Regime = $vol
            PredH2 = $x.H2.selected
            PrecisionH2Pct = [math]::Round(100 * $x.H2.pooled_precision, 2)
            UpliftH2Points = [math]::Round($x.H2.pooled_uplift_points, 2)
            DeltaUpliftMoyen = [math]::Round($x.paired.H2_minus_H6_uplift_points_mean, 2)
        }
    }
}
$actifs | Format-Table -AutoSize
\`\`\`

Pour analyser les 24 heures : \`$r.summary.hour_utc\` contient des cellules heure × actif, avec effectifs pour éviter les conclusions à partir de petits sous-échantillons.

**Avertissement méthodologique** : la référence aléatoire utilise la proportion des classes effectivement observées dans la cellule pour mesurer l'uplift descriptif, ce n'est pas une information disponible ex ante en production. Tous les checks sont des validations P2 déjà explorées.
