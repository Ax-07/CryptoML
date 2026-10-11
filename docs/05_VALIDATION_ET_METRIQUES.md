# 05 — Walk-forward, généralisation et métriques

## Pourquoi deux axes de validation ?

- **Généralisation temporelle** : les mêmes paires sur des périodes ultérieures inconnues.
- **Généralisation cross-asset** : des paires intégralement exclues du **développement** du modèle et des réglages.

Les deux sont nécessaires pour soutenir la promesse « adaptable à une nouvelle paire ». Cela ne garantit pas que toutes les paires futures seront prévisibles.

## Protocole recommandé

1. Fixer dès le début un **test final temporel** (derniers 3–6 mois selon couverture et besoin) et **un groupe de paires test final** inconnu du train/validation.
2. Sur le reste des dates et paires autorisées : walk-forward glissant (exemple initial 120 jours train, 14 jours validation, 14 jours test OOS, pas de 14 jours). Les fenêtres sont des paramètres expérimentaux, pas des vérités absolues.
3. Les paramètres et le seuil de label sont sélectionnés exclusivement au moyen du train et de la validation des folds de développement.
4. Avant tout split : pour un sample (symbole, temps t, horizon H), calculer `label_mature_at`. L'exclure du train si `label_mature_at >= validation_start`. Appliquer l'équivalent à la séparation validation → test ; vérifier les dépendances temporelles des variables.
5. Préserver des intervalles de **test chronologiquement disjoints** afin de ne pas compter plusieurs fois les mêmes événements.
6. Produire métriques **par fold, horizon, crypto, classe et régime**, puis agrégations robustes.
7. Utiliser le test final **une fois après gel des choix**, et archiver le rapport.

### Attention aux corrélations

BTC/ETH/SOL peuvent réagir au même événement ; des millions d'échantillons M5 chevauchants ne sont pas indépendants. Estimer l'incertitude avec des méthodes temporelles (ex. bootstrap par blocs ou par périodes) et rapporter nombres de journées/semaines et paires, pas seulement lignes.

## Métriques principales (3 classes)

| Métrique | Pourquoi | Précaution |
|---|---|---|
| Balanced Accuracy | Rappel moyen des classes | Plus informative qu'accuracy brute si classes inégales |
| Macro F1 | Précision/rappel équilibrés | Reporter support des classes |
| Matrice de confusion | Erreurs UP/NEUTRAL/DOWN | Réel en lignes, prédit en colonnes ; effectifs + pourcentages |
| Log loss | Qualité probabiliste multiclasses | Sensible à la surconfiance |
| Brier multiclasses | Écart au vecteur one-hot réel | Bien définir la convention de somme des carrés |
| Calibration | Fiabilité des probabilités | Courbes par classe + effectifs des bins |
| Performance par crypto/horizon | Robustesse | Afficher tailles d'échantillon et période |

En complément : accuracy, rappel UP/DOWN, distribution des probabilités, quantiles d'incertitude et sensibilité au seuil neutre. **Éviter AUC binaire seule comme métrique du modèle multiclasses** ; si AUC OVR, expliquer la méthode.

## Baselines, indispensables

- Classe majoritaire apprise sur train uniquement.
- Persistance directionnelle simple (mouvement récent → prévision selon une règle fixe et publique).
- Régression logistique multiclasses régularisée.
- Pour l'accuracy, tenir compte de la proportion des classes ; **33 % n'est pas nécessairement la référence du dataset**.

## Seuils de validation

Ne pas fixer un « score magique » de succès. Un modèle est candidat s'il apporte une amélioration OOS reproductible vs baselines, que la calibration est acceptable, que les performances par paire/régime ne s'effondrent pas et que l'incertitude est documentée. Une performance faible doit pouvoir être acceptée comme résultat scientifique.

## Rapports de validation

- `metrics.json` : métriques agrégées et par fold/crypto/classe.
- `confusion_matrix.json` : matrice et supports.
- `calibration.json` : calibration par classe/bin + tailles.
- `oos_predictions.parquet` : prédictions **au moment du fold**, sans fuite.
- `folds_manifest.json` : dates train/validation/test et exclusions, paires.
- `comparisons.json` : baselines, horizons, modes avec/sans contexte BTC/ETH.

La page « Qualité prédictive » doit séparer visuellement **train**, **validation**, **walk-forward OOS**, **test final temporel**, et **paires inédites**.
