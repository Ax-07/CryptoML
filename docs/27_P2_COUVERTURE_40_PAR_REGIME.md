# CryptoML P2 — Couverture cible 40 % par régime, H2 vs H6

## Protocole

Ce protocole recherche si les avantages exploratoires de H2 (10 minutes) sur H6 (30 minutes) restent visibles **quand la couverture directionnelle est ciblée à 40 % sur le buffer de tuning**, par régime. Ne jamais prétendre que la couverture constatée sur le check est identique.

- 36 folds de développement du même archive de 24 mois. Purge de maturité **H12=60 minutes** commune. 120 jours train, 7 jours early stop, 3 jours tuning, 4 jours check, 14 jours OOS non notés, 90 jours finaux non notés.
- Deux LightGBM CPU multiclasse complets par fold, 30 features, alpha=1.0. 72 entraînements, sans relecture d'ADA ; pas de nouveau modèle exporté.
- Régimes préspécifiés indépendamment : 4 blocs UTC de six heures, semaine/week-end, terciles de \`volatility_48\` par actif. Les seuils des terciles utilisent **exclusivement le train** ; les groupes sont attribués à partir de l'heure et de la volatilité de la bougie courante fermée.
- Pour chaque \`horizon × fold × axe × régime\`, choisir le seuil sur **les prédictions sans labels du seul buffer 3 jours**, à couverture cible 40 % et avec la fonction historique \`threshold_for_target\`. Pas de réglage d'après labels du check.
- Un régime avec moins de 100 lignes de tuning ou trop peu de prédictions directionnelles reste **infeasible** : aucune substitution cachée d'un seuil global. Un week-end peut manquer d'observations dans une fenêtre de 3 jours, c'est normal.
- Les comparaisons de checks excluent les paires avec moins de 20 prédictions directionnelles par horizon ; les effectifs et nombres de folds exclus sont enregistrés explicitement. Les comparaisons par actif réutilisent les mêmes seuils globaux par régime, sans garantie de couverture 40 % par actif.
- Sorties : précision, couverture **réalisée**, uplift vs référence conditionnée à la fréquence des classes du régime+fold, erreurs NEUTRAL, supports et comparaison appariée H2–H6 par axe/régime et crypto. **Un uplift positif n'est pas un retour de trading**.
- Pas de filtre heure × volatilité combiné pour éviter une exploration supplémentaire des sous-groupes rares.
- Attention : H2 et H6 ont des labels futurs différents, validations déjà largement observées, folds corrélés, sélection adaptative des régimes. Il s'agit seulement d'une analyse descriptive, **pas de confirmation OOS indépendante**.

## Exécution Windows PowerShell

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q
\`\`\`

Si tests verts :

\`\`\`powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.regime_coverage_matched \`
  --data-root "E:\cryptoML\data" \`
  --months 24 \`
  --n-estimators 150 \`
  --verify-hashes \`
  --output "E:\cryptoML-walkforward-results\regime_coverage40_h2_h6_p2.json"
\`\`\`

Le rapport final n'écrase jamais les anciens résultats.

## Résumé à copier dans le dialogue

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\regime_coverage40_h2_h6_p2.json" -Raw | ConvertFrom-Json
$comparaison = foreach ($axe in @("utc_block_6h","day_type","volatility_48_train_tercile")) {
    foreach ($prop in $r.summary.$axe.PSObject.Properties) {
        $x = $prop.Value
        [PSCustomObject]@{
            Axe = $axe
            Regime = $prop.Name
            FoldsApparies = $x.paired_folds_eligible_min20
            FoldsInfeasiblesOuTropPetits = $x.excluded_tuning_or_small_check_folds
            H2MeilleurUplift = $x.h2_better_uplift_folds
            PrecisionH2Pct = [math]::Round(100 * $x.horizons."2".precision_pooled,2)
            PrecisionH6Pct = [math]::Round(100 * $x.horizons."6".precision_pooled,2)
            UpliftDeltaPoints = [math]::Round($x.uplift_delta_points_fold_mean,3)
            CouvertureH2Pct = [math]::Round(100 * $x.horizons."2".coverage_pooled,2)
            CouvertureH6Pct = [math]::Round(100 * $x.horizons."6".coverage_pooled,2)
            EcartCouverturePoints = [math]::Round($x.realized_coverage_delta_points_fold_mean,3)
        }
    }
}
$comparaison | Format-Table -AutoSize
\`\`\`

**Lecture** : seules les fenêtres appariées et éligibles figurent dans les moyennes, pas nécessairement les 36. Les écarts de couverture sur checks peuvent être importants : ils doivent être interprétés conjointement aux écarts de précision et d'uplift. Les catégories non faisables ne doivent pas être présentées comme zéro gain.
