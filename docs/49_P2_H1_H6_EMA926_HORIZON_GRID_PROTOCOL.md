# P2 — Grille H1–H6 : EMA 9/26 minimal vs LightGBM 30 (protocole pré-spécifié)

**Statut : hypothèse exploratoire, répétition de checks historiques déjà inspectés. Aucune confirmation indépendante.**
**Date : 2026-10-19. Branche : feat/cpu-purged-walkforward-folds.**

## Objectif et vérité à prédire

À chaque **clôture M5 t**, produire six ensembles de probabilités \`P(DOWN), P(NEUTRAL), P(UP)\` pour les **cours de clôture t+H**, H=1..6 (5 à 30 minutes). Les horizons H2-H6 ne sont **pas** la couleur de la H-ième bougie : ils représentent les rendements cumulés t→t+H.

L'objectif **prioritaire H1** comporte **deux vérités différentes**, à toujours publier côte à côte :
- **Label canonique 3 classes** : \`UP\` si \`r(t→t+H)>0.5*volatility_48(t)*sqrt(H)\`, \`DOWN\` si l'opposé, sinon \`NEUTRAL\`. La volatilité est passée seulement, et le futur doit être exact et continu.
- **Signe brut de la bougie suivante** : \`UP\` si \`close(t+1)>close(t)\`, \`DOWN\` si inférieur, \`FLAT\` si égal. Pour la précision du **signe** exclure les rares \`FLAT\` du dénominateur et publier leur nombre. **Aucune valeur absolue de seuil**, ni détournement du label ternaire.

## Une et une seule expérience pré-déclarée

| Modèle | Entrées | Paramètres |
|---|---|---|
| \`lightgbm_30\` | Les 30 colonnes \`features_v1\`, ordre inchangé | LightGBM 3 classes CPU, 150 estimateurs maximum, seed 42, early stopping 30 |
| \`lightgbm_ema2\` | \`ema9_price_gap=close/EMA9-1\` et \`ema9_ema26_gap=EMA9/EMA26-1\` | **Même** algorithme et paramètres |
| \`ema_cross_rule\` | Signe \`EMA9-EMA26\` (équivaut au signe de \`ema9_ema26_gap\`), égalité = majorité directionnelle issue du train | Aucun apprentissage |

Les EMA utilisent les bougies M5 **clôturées jusqu'à t**, \`ewm(span=9/26,adjust=False,min_periods=9/26)\`, et réinitialisent leurs fenêtres à chaque interruption de continuité M5. Jamais de prix brut non normalisé, d'asset-ID ni de label dans les inputs. Les périodes 9/26 sont figées ; aucune recherche de 8/21, 12/26 etc. sur les checks.

Les horizons H1..H6 sont des **modèles séparés** : six paires de LightGBM, pas un réseau neuronal multi-sorties. Le vecteur de six horizons peut être reconstruit à chaque timestamp sans information du futur.

## Comparaisons explicites et séparées

1. \`all_check\` : toutes les lignes, signes UP/DOWN imposés par comparaison des **probabilités brutes** \`pUP>pDOWN\` (égalité => majorité train). NEUTRAL compte erreur. C'est la comparaison réellement appariée sur les six horizons.
2. \`shared_baseline_selected\` : seules les lignes choisies par le modèle **30 features** à partir d'un seuil d'abstention ajusté aux **scores du tuning de 3 jours sans labels** (cible 40 %, alpha=1). Tous les signaux sont évalués sur **le même masque** (une règle simple n'y est pas autonome).
3. \`minimal_autonomous_selected\` : le modèle EMA2 sélectionne indépendamment sur ses propres scores de tuning (40 % cible) ; sa précision/couverture est **descriptive et non directement appariée** avec celle du baseline sur des lignes différentes.
4. \`ema_rule_autonomous_selected\` : la règle EMA sélectionne via un seuil sur \`abs(ema9_ema26_gap)\` déterminé sur les valeurs **sans labels** du tuning précédent (40 % cible). Comparaison de sélecteurs différents = non appariée.
5. **H1 next-bar sign** : précision des prédictions directionnelles sur la direction brute \`close(t+1)-close(t)\`, uniquement pour les variations strictement non nulles ; publier \`flat\`, taux d'erreurs et couverture. Pas d'usage du signe comme entrée.
6. Diagnostics multiclasses secondaires (log loss / Brier), classe observée, détection de mouvement, sans optimization de seuil.

**Ne jamais confondre** \`pUP>pDOWN\` avec l'argmax à 3 classes : le signe forcé peut prédire UP alors que NEUTRAL a la probabilité maximale. Les marges alpha=1 servent exclusivement à la sélection de couverture.

## Découpage, contrôles et stockage

- Reprendre les **36 folds** du plan P2 original : 120 jours train, 7 jours early stop, 3 jours tuning scores sans labels, 4 jours development check ; purge de maturité **H12=60 min** pour aligner tous les horizons ; environ **12 fits par fold**, **432 fits CPU** en complet.
- Couper chaque actif **avant la fin du check propre à chaque fold** AVANT toute construction des labels. Aucune ligne des OOS des folds, dernier holdout 90 jours ou ADA ne peut être scorée ; pas de modèles P3 ou H6 forward modifiés.
- Reproduire **exactement** le benchmark historique \`p2_directional_rule_benchmark_full_v1.json\` pour H2/H6 (nombre de lignes, comptes corrects de la baseline 30, masque/itération/partition) ; fail closed si divergences. Pour H1, H3, H4, H5 ne pas inventer une nouvelle géométrie : toutes les lignes devront avoir les six labels.
- Empreinte SHA dataset/manifests validée avant tout scoring. Le rapport JSON et le fichier Parquet des **prédictions de checks historiques** restent exclusivement dans \`E:\cryptoML-walkforward-results\`, nouveaux noms et **sans écrasement**. Le parquet a une ligne par (fold, actif, t, horizon), probabilités des **deux** LightGBM, prédictions signées, masks de sélection, vraie classe mature et retour futur (uniquement pour l'évaluation historique). Ni prédictions train/tuning ni observations OOS réservées.
- Tests synthétiques : EMA causal préfixe/future-blind, gaps M5, échelle de prix, labels H1..H6 exacts et maturité H12, invariance des lignes, appariement, abstention autonome, reproduction historique H2/H6, n'importe quel drift => arrêt.
- **Pas de nouvelle optimisation sur ces checks** après les résultats ; les horizons peuvent être décrits mais **ne deviennent pas six nouvelles validations indépendantes**.

## Sortie attendue / décision

Le tableau final présente, **pour chacun de H1..H6** : effectifs, taux de NEUTRAL, précision 30 et EMA2 et règle EMA en all-check et sur sélection commune, précisions autonomes et leurs couvertures réalisées, gain apparié moyen par fold et agrégé, nombre de folds positifs/négatifs, logloss et Brier. H1 ajoute la prédiction du **signe brut de la prochaine bougie** (FLAT exclu de la précision mais dénombré). Le Parquet permet de revoir **toutes les prédictions** par bougie, actif, modèle et horizon sur les checks exploratoires.

**Pas de promotion automatique** même si H1 paraît favorable. Les résultats historiques déjà examinés n'ont qu'une valeur exploratoire ; une confirmation chronologiquement nouvelle, préenregistrée et autorisée est nécessaire. Aucune estimation de frais, slippage, PnL ou rentabilité.
