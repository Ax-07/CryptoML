# P2 — Décomposer la prédictibilité H2/H6 avant toute nouvelle architecture

## Question

Sur les checks historiques de développement déjà inspectés, le modèle LightGBM **30 features**, label fixe `vol_scaled_v1` multiplicateur 0,5, est-il meilleur pour :
- A : classer le fait que la cible à horizon fixe soit un mouvement (DOWN/UP) plutôt que NEUTRAL ?
- B : parmi les vrais mouvements **connus seulement rétrospectivement**, distinguer UP de DOWN ?
- C : est-ce différent selon le niveau de volatilité **visible à la bougie t** ?

Cette étude ne modifie **ni labels, ni features, ni modèles figés**. Elle réentraîne uniquement la référence 3 classes **dans une expérience de diagnostic isolée**, 150 estimateurs CPU, seed 42. Aucune recherche d'hyperparamètres ou de seuils, ni comparaison de variantes.

## Protocole strict

- Le plan et les effectifs sont lus du JSON d'ablation P2 préexistant ; 36 folds, 120d entraînement, 7d early stopping, 3d tuning ignorés, 4d validation-check descriptifs et maturité H12 (60m).
- Les quatre archives BTC/ETH/SOL/XRP sont lues en mode consultation. **Les bougies sont tronquées à la dernière limite de check incluse AVANT la création des labels.** Aucun label des tests OOS des folds, des 90 derniers jours finaux ou d'ADA n'entre dans l'analyse. Les checks peuvent néanmoins correspondre à des périodes qui se retrouvent dans l'entraînement de folds ultérieurs ; ces folds ne constituent pas des confirmations indépendantes.
- A = probabilité multiclasses brute `1-P(NEUTRAL)`; métriques average precision (AP), ROC-AUC, Brier, logloss ; référence Brier/logloss = fréquence de mouvement de l'entraînement **du même fold**. La prévalence sur le check n'est qu'un repère descriptif pour AP, pas un seuil ajusté.
- B = `P(UP)/(P(UP)+P(DOWN))`, mesurée seulement sur les lignes réellement DOWN/UP **en regardant les labels futurs**. C'est une mesure **oracle conditionnelle** qui ne peut pas être utilisée en inférence comme un filtre idéal. Référence = proportion UP parmi les vrais mouvements de l'entraînement.
- C = `LOW/MID/HIGH` selon les quantiles 1/3 et 2/3 de `volatility_48`, **pour chaque actif et à partir de ses seules lignes d'entraînement**. La métrique par régime a comme référence les étiquettes d'entraînement du même régime et du même actif. Ne pas interpréter la meilleure sous-population a posteriori comme une opportunité confirmée.
- Les métriques sont calculées sur les probabilités multiclasses **brutes**, pas sur les marges corrigées par le prior `alpha=1` ; la comparaison avec l'uplift directionnel à couverture 40 % n'est donc pas directe.
- Les checks sont déjà inspectés, les folds et labels à horizons proches sont dépendants. Résultats **exploratoires uniquement** : pas d'affirmation de rentabilité ni d'amélioration de performance hors échantillon indépendante.

## Commandes — checkout Windows isolé

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Smoke 1 fold = 2 entraînements de 3 classes (H2/H6) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.predictability_decomposition `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_predictability_h2_h6_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

Après tests et smoke réussis, rapport complet 36 folds = **72 entraînements** :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.predictability_decomposition `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_predictability_h2_h6_full_v1.json" `
  --months 24 --n-estimators 150 --seed 42 --verify-hashes
```

Le rapport JSON fournit les métriques par horizon, fold, actif et régime. Les fichiers de sortie existants ne sont pas remplacés.

Résumé PowerShell :

```powershell
$r=Get-Content "E:\cryptoML-walkforward-results\p2_predictability_h2_h6_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($h in @("H2","H6")) {
  $s=$r.summary.$h
  [PSCustomObject]@{
    Horizon=$h
    Movement_AP_Excess_Points=[math]::Round($s.movement.mean_ap_excess_over_check_prevalence_points,3)
    Movement_Brier_Skill=[math]::Round($s.movement.mean_brier_skill_vs_train_prior,5)
    Movement_ROC_AUC=[math]::Round($s.movement.mean_roc_auc,4)
    Direction_Oracle_Accuracy_Gain_Points=[math]::Round($s.direction_oracle.mean_accuracy_delta_points,3)
    Direction_Oracle_Brier_Skill=[math]::Round($s.direction_oracle.mean_brier_skill_vs_train_prior,5)
    Direction_Oracle_ROC_AUC=[math]::Round($s.direction_oracle.mean_roc_auc,4)
  }
} | Format-Table -AutoSize
```

En PowerShell 5.1, si un pipeline directement après `foreach` n'est pas accepté, entourer `@(foreach (...) {...})` ou assigner `$rows = foreach (...) {...}` et exécuter `$rows | Format-Table -AutoSize`.

P3 reste arrêté ; aucun autre environnement Python ne doit être activé et aucune commande de production n'est exécutée.
