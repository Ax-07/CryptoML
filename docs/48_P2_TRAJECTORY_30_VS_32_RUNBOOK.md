# P2 — Exécuter la comparaison appariée 30 vs 32 features (protocole 47)

**Statut : implémentation expérimentale, tests et smoke Windows à réaliser avant tout run complet.**  
Module : \`backend/mlpipeline/trajectory_direction_comparison.py\`  
Hypothèse / seuils **déjà figés** : \`docs/47_P2_PREDECLARED_TRAJECTORY_DIRECTION_RESEARCH.md\`.

L'essai compare, aux horizons H2/H6, le LightGBM original de 30 features à une **unique** variante à 32 features : \`trajectory_signed_efficiency_12\` et \`trajectory_prior_range_position_12\`. Il n'altère pas \`features_v1\`, les labels ni les modèles P3.

Le masque de sélection des décisions est **exclusivement** celui du LightGBM 30, calibré en couverture ~40 % à partir des scores du *tuning antérieur*, sans ses labels ; les deux modèles sont jugés sur les mêmes lignes et chaque NEUTRAL compte comme erreur. La réversion n'est qu'un diagnostic annexe. Les performances probabilistes brutes (logloss, Brier, mouvement AP/AUC, direction oracle) sont enregistrées à part sur tous les checks. Les 36 checks de développement ont déjà été observés : **pas de validation indépendante**.

Les données P0 sont consultées en lecture seule, avec empreinte ; avant création du moindre label, chaque fold tronque à **sa** fin de check. L'alignement des 30/32 colonnes interdit la suppression silencieuse d'une seule ligne. Chaque fold refait les comptes du LightGBM 30 et exige leur **égalité exacte** avec le rapport initial \`p2_directional_rule_benchmark_full_v1.json\`. Le rapport est créé uniquement si l'exécution entière aboutit, en refusant d'écraser un fichier existant. ADA, tests OOS de folds, 90 derniers jours, production, P3, données live et tout trading sont exclus.

## 1. Mettre à jour le checkout isolé et exécuter TOUS les tests

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_trajectory_direction_features.py tests/test_trajectory_direction_comparison.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

En cas d'erreur, **arrêter** avant tout entraînement historique.

## 2. Smoke technique 1 fold (2 modèles x H2/H6 = 4 fits CPU)

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.trajectory_direction_comparison \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" \`
  --output "E:\cryptoML-walkforward-results\p2_trajectory_30vs32_smoke_v1.json" \`
  --months 24 --max-folds 1 --verify-hashes
\`\`\`

Un résultat de smoke est **uniquement technique**, sans valeur scientifique de comparaison. S'il échoue : arrêter, ne pas affaiblir les assertions des effectifs/comptes/folds.

## 3. Full : 36 folds × 2 horizons × 2 modèles = 144 fits CPU

**N'exécuter que si les tests unitaires et le smoke sont verts.**

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.trajectory_direction_comparison \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" \`
  --output "E:\cryptoML-walkforward-results\p2_trajectory_30vs32_full_v1.json" \`
  --months 24 --verify-hashes
\`\`\`

## 4. Lire le résumé

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_trajectory_30vs32_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = foreach ($h in @("H2", "H6")) {
    $s = $r.summary.$h
    [PSCustomObject]@{
        Horizon = $h
        Folds = $s.comparable_folds
        Precision30Pct = [math]::Round(100*$s.mean_baseline_precision, 3)
        Precision32Pct = [math]::Round(100*$s.mean_candidate_precision, 3)
        DeltaMoyenPts = [math]::Round($s.mean_gain_points, 3)
        DeltaPonderePts = [math]::Round($s.pooled_gain_points, 3)
        Folds32Meilleur = $s.folds_candidate_wins
        Folds30Meilleur = $s.folds_baseline_wins
        Logloss32Moins30 = [math]::Round($s.mean_logloss_candidate_minus_baseline, 5)
    }
}
$rows | Format-Table -AutoSize
$r.summary.predeclared_gates | Format-List
$r.summary.decision
\`\`\`

La décision restera **exploratoire** même si tous les critères préspécifiés passent. La comparaison \`shared_baseline_selected\` diffère d'une stratégie 32 features sélectionnant ses propres signaux. Si le gain manque le seuil fixé, ne pas entreprendre une recherche de 6/24 barres ou de nouvelles variantes sur les mêmes checks dans le seul but d'augmenter le score.

**Ne jamais activer \`E:\cryptoML\backend\.venv\Scripts\activate\` :** appeler toujours directement le Python isolé ci-dessus.
