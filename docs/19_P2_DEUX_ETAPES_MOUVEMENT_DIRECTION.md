# CryptoML P2 — Comparatif LightGBM 3 classes / 2 étapes

## Hypothèse

Les 55 372 décisions filtrées en validation de développement étaient : 33,62 % correctes, 37,99 % sur des mouvements NEUTRAL, 28,39 % dans la direction contraire. Ce diagnostic suggère de **tester**, sans présumer du résultat, la séparation des problèmes mouvement / direction.

## Protocole

Le script `mlpipeline.two_stage_diagnostics` entraîne les **mêmes trente features**, en CPU, dans les **mêmes folds 120/14/14/14 jours**, avec les mêmes paramètres LightGBM généraux et la même définition des labels H6 (30 minutes) que le modèle de référence.

Une validation de 14 jours est subdivisée ainsi :
- jours 1–7 : arrêt anticipé des trois modèles ;
- jours 8–10 : buffer chronologique **non utilisé** dans ce comparatif (aucun seuil optimisé) ;
- jours 11–14 : contrôle comparatif des deux méthodes, sur les mêmes observations.

**Référence** : LightGBM multiclasse DOWN/NEUTRAL/UP.

**Méthode en deux étapes** :
1. Modèle binaire `P(mouvement significatif | X)`, entraîné sur toutes les classes, cible `target != NEUTRAL`.
2. Modèle binaire `P(UP | mouvement significatif, X)`, entraîné uniquement sur les exemples DOWN/UP de la période **d'entraînement et d'early stopping**.

Les deux sorties sont composées **sur chaque observation, sans jamais connaître sa classe future** :

```text
P(DOWN)    = P(mouvement) * (1 - P(UP | mouvement))
P(NEUTRAL) = 1 - P(mouvement)
P(UP)      = P(mouvement) * P(UP | mouvement)
```

Ces trois nombres forment des probabilités dont la somme vaut 1. Les mêmes décisions correctrices fixes `alpha=1.0` sont appliquées aux deux modèles **uniquement pour les décisions**, pas pour les scores probabilistes.

Comparer sur chaque fenêtre :
- précision UP/DOWN quand prédits (NEUTRAL compte comme erreur) et couverture ;
- Macro-F1, balanced accuracy des décisions `alpha=1.0` ;
- log loss et Brier **sur probabilités brutes** ;
- types d'erreurs (NEUTRAL ou direction opposée), résultats par BTC/ETH/SOL/XRP ;
- AUC/accuracy de la détection des mouvements (diagnostic descriptif) ;
- qualité de la deuxième étape quand un mouvement s'est réellement produit (**diagnostic oracle descriptif uniquement**, ne pas l'interpréter comme taux de réussite réel).

**Important** : les validations de développement ont déjà été inspectées dans P1/P2, et ne constituent PAS une validation indépendante. Aucun fold OOS n'est évalué, ni les 90 derniers jours, ni ADA. Aucun modèle de production, export figé ou signal live n'est modifié. Les résultats sont uniquement des hypothèses à tester plus tard sur des données prospectives.

## Exécution PowerShell

Après récupération de la branche et validation des tests :

```powershell
cd E:\cryptoML-walkforward-test\backend

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.two_stage_diagnostics `
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
  --output "E:\cryptoML-walkforward-results\two_stage_p2_24m.json"
```

Le rapport est écrit exclusivement à la destination donnée et refuse tout écrasement. **Ne pas réutiliser les résultats finaux consultés pour sélectionner le vainqueur.**

## Analyse des résultats

Lire la section JSON `fold_mean_summary` et comparer `three_class` et `two_stage`. Une hausse de précision avec une couverture beaucoup plus faible n'est pas nécessairement une amélioration ; un meilleur F1 avec une dégradation importante de log loss ne suffit pas non plus. Examiner les cas défavorables, par fold et par actif, avant toute conclusion.

Si les résultats sont décevants, revenir aux labels/features dans une **nouvelle expérience de développement**. La validation finale de tout candidat P2 nécessite des données vraiment nouvelles après les périodes déjà ouvertes.
