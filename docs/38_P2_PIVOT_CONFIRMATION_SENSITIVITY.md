# P2 historique — sensibilité de confirmation des pivots R1/R2/R3

Objectif : déterminer si le délai de confirmation des swings explique le faible gain du précédent modèle 42 features. **R1, R2 et R3 désignent ici 1, 2 ou 3 bougies M5 de confirmation : ce n'est PAS le protocole prospectif P3.**

Les pivots high/low reposent toujours sur deux bougies **antérieures**. Un sommet potentiel au temps p est comparé strictement aux highs antérieurs p−2 et p−1 et aux highs suivants p+1 à p+r ; il n'est donc connu et utilisé qu'à **p+r**, avec chaque bougie clôturée. Les égalités ne comptent pas. Les gaps M5 réinitialisent la structure. L'ancien calcul R2 est conservé comme valeur par défaut dans `market_structure_features.py`.

## Modèles comparés

- `baseline_30` : 30 features inchangées
- `structure_r1_42` : les 30 + 12 de structure confirmées sur 1 bougie
- `structure_r2_42` : les 30 + 12 déjà étudiées (2 bougies)
- `structure_r3_42` : les 30 + 12 confirmées sur 3 bougies

Même ensemble de lignes pour les quatre versions, excluant toutes les warmups invalides. Pour chaque horizon H2/H6 : train 120j, validation 7j early-stop, tuning 3j sur **scores seuls** pour une couverture cible de 40 %, vérification 4j. Purge H12 avant chaque frontière, aucun test OOS de fold, ADA ni dernier holdout 90 jours. La couverture réalisée sur la vérification et la non-faisabilité sont toujours rapportées. Les dates de vérification ont déjà été explorées : **aucune validation indépendante n'est revendiquée**. Aucun modèle P3 figé ni collecte prospective n'est modifié.

## PowerShell

Depuis le worktree historique :

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q tests/test_market_structure_confirmation_sensitivity.py
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Smoke de 8 entraînements (1 fold × H2/H6 × 4 variantes) :**

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.market_structure_confirmation_sensitivity `
  --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_pivot_confirmation_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

**Tous les folds, après smoke réussi :** retirer `--max-folds 1` et utiliser `--output "E:\cryptoML-walkforward-results\p2_pivot_confirmation_full_v1.json"`. Le plan complet comporte au plus **288 entraînements CPU (36 × 2 × 4)**. Un fichier existant n'est jamais écrasé.

Le JSON conserve `summary.H2/H6.paired_vs_baseline` : delta d'uplift, proportion de folds favorables, évolution de la couverture réalisée et du log loss. Les fréquences d'événements sur les mêmes checks se trouvent dans `check_event_frequency`, globales et par actif.

Le code est déposé sur GitHub mais les tests ne peuvent être considérés comme réussis avant leur exécution sur l'environnement local Windows.
