# CryptoML P2 — Comparaison à couverture cible identique (CPU)

## Objectif

Le précédent comparatif a montré +0,683 point de précision directionnelle moyenne pour le modèle à deux étapes, mais avec une couverture moyenne inférieure de 4,15 points. Ce nouveau protocole compare les deux modèles à **couvertures visées identiques sur une période passée**, sans forcer artificiellement la couverture des périodes ultérieures.

## Protocole pré-déclaré

- **Cible principale de couverture : 30 %** des observations M5.
- **Cible secondaire : 40 %**.
- 120 jours train ; 7 jours early stopping ; 3 jours tuning des seuils par modèle ; 4 jours validation check.
- Mêmes 30 features et labels H6 ; CPU ; alpha fixé à 1.0 ; modèle LightGBM 3 classes vs modèles LightGBM binaires mouvement/direction.
- Sur les 3 jours tuning, chaque modèle calcule des scores de décision UP/DOWN et une marge top1-top2. Le seuil de marge est choisi **sans lire les labels du tuning** afin d'approcher la couverture visée.
- Sur les 4 jours check, le seuil est **figé**. Les couvertures réelles peuvent diverger, et cette divergence doit être reportée.
- S'il y a trop peu de décisions UP/DOWN en tuning pour atteindre la cible, ou si les ex aequo empêchent d'approcher la cible à 2 points près, cette fenêtre est déclarée non comparable pour la cible concernée. Ces exclusions sont déterminées **sans les labels check**. Une couverture check nulle entraîne une précision non définie.
- Les deux architectures sont entraînées uniquement avec labels arrivés avant chaque frontière temporelle. La cible étape 2 est UP/DOWN uniquement sur les mouvements effectifs **de la période d'entraînement**, jamais sur les observations check.

Le script rapporte précision directionnelle (NEUTRAL est une erreur), couverture, taille des groupes, uplift contre références aléatoires globales et conditionnelles, différences par fold, et résumés de paires comparables. **Une cible de couverture identique sur tuning ne garantit pas une couverture identique sur check.** Aucune adaptation ex post n'est permise.

## Sécurité d'évaluation

- Les OOS des folds, la période finale de 90 jours et ADA ne sont pas évalués.
- Ces fenêtres de validation ont déjà été examinées lors des recherches précédentes : **ce n'est pas un nouveau test indépendant**.
- Aucun artefact de trading, ordre, modèle figé ou donnée historique n'est modifié.
- Le JSON refuse tout écrasement.

## Windows PowerShell

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.coverage_matched_diagnostics `
  --data-root "E:\cryptoML\data" `
  --months 24 `
  --horizon 6 `
  --train-days 120 `
  --validation-days 14 `
  --test-days 14 `
  --step-days 14 `
  --final-holdout-days 90 `
  --n-estimators 150 `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\matched_coverage_p2_24m.json"
```

Pour les résumés :

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\matched_coverage_p2_24m.json" -Raw | ConvertFrom-Json
$r.summary | ConvertTo-Json -Depth 8
```

Les résultats doivent être interprétés avec l'écart de couverture **réellement observé** sur les quatre jours check et les folds exclus selon faisabilité du tuning. Pour une véritable confirmation, une évaluation prospective sur de nouvelles observations est indispensable.
