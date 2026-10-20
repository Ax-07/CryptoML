# P2 — Étude historique exploratoire de structure HH/HL/LH/LL

**Statut :** code de recherche prêt à être testé localement, aucun résultat de marché encore calculé. Pas de trading, aucun horodatage tiers, pas d'exécution en temps réel. Ne pas altérer les modèles H2/H6 déjà figés ni l'inscription personnelle P3.

## Hypothèse

Une représentation explicitement causale de la structure des sommets et creux peut améliorer H2 (10 min) et/ou H6 (30 min) au-delà des 30 features `features_v1`. Les validations de développement ont déjà été examinées ; **tous les nouveaux résultats seront exploratoires, non indépendants**.

## Définition mécanique V1

Module `mlpipeline.market_structure_features`, 12 colonnes :
- `structure_hh`, `structure_hl`, `structure_lh`, `structure_ll` : comparaison du dernier swing confirmé au précédent de même type. Les états persistent jusqu'à confirmation suivante.
- `structure_bos_up/down` : franchissement en clôture d'un précédent pivot connu, avec HH+HL ou LH+LL confirmé précédemment.
- `structure_choch_up/down` : même franchissement, mais contre la structure confirmée précédemment.
- `structure_distance_swing_high/low` : distance relative au dernier pivot confirmé.
- `structure_age_swing_high/low` : ancienneté de confirmation / 48 barres.

Un swing high au temps `p` est strictement supérieur aux highs `p-2,p-1,p+1,p+2`, le swing low est strictement inférieur aux quatre lows. **La feature du pivot n'est visible qu'à `p+2`**, après la fermeture de ces deux barres. Les égalités sont ignorées. Pas d'accès au futur, pas de calcul sur bougie ouverte, aucune continuité fabriquée lors d'un gap M5. Les BOS/CHoCH ne prétendent pas être les définitions discrétionnaires universelles du trading.

## Variantes appariées

- `baseline_30` : les 30 features inchangées ;
- `baseline_plus_structure_42` : 30 + 12 nouvelles ;
- `structure_only_12` : structure seule.

Même lignes éligibles et mêmes folds H12-purgés pour les trois. Pour chaque fold : 120 jours train, 7 jours early stopping, 3 jours de tuning **sans labels utilisés dans le choix de seuil**, puis 4 jours de vérification descriptive. La couverture **cible** est 40% sur tuning ; la couverture réellement obtenue sur check est mesurée séparément. En cas de seuil irréalisable, le fold est signalé et exclu de la comparaison appariée (sans inventer de score). H2/H6 ont des cibles différentes, leurs scores ne sont pas une mesure de profit.

## Démarrage local, sans toucher à P3

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q tests/test_market_structure_features.py
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Si les tests sont verts, calcul historique CPU (peut être lourd, 36 folds × 2 horizons × 3 variantes = jusqu'à 216 entraînements) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.market_structure_diagnostics `
  --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_market_structure_h2_h6_v1.json" `
  --months 24 --n-estimators 150 --seed 42 --verify-hashes
```

Le programme refuse d'écraser un rapport existant. Tous les calculs et résultats restent locaux ; l'assistant ne peut ni lancer ce processus CPU sur ton PC ni annoncer le résultat avant réception du rapport.

Résumé à examiner : `report["summary"]["H2"]` et `report["summary"]["H6"]`, surtout `paired_vs_baseline` : `mean_uplift_delta_points`, `positive_uplift_folds`, `mean_realized_coverage_delta_points`, `mean_log_loss_improvement`, ainsi que `paired_feasible_folds` pour éviter une interprétation trompeuse.
