# P4 V39 — Proposition de contrat de cible, SANS création de labels

## État d'entrée

Le rapport V38 fourni par l'opérateur atteste 33 folds théoriques ; SHA du rapport V35 `ad478769e6b3d58f3df7143fb8d5f917937a3f10eba090f5747700a2cd8e2abe`, SHA V37 `33308282142445e912faa1a8ac215e38e0e9764d5ff97f7e64859410f1235d08`, gel V33 `36983616ac8a5424ac71fd4be3ba3ccb7885becf481052dde431a2951a12ae71`.

Les comptes agrégés V38 sont 4 560 804 créneaux train, 530 640 validation, 530 640 test ; **ces nombres sont des bornes théoriques répétées entre folds, pas des labels ni des exemples uniques.**

## Spécification proposée (non approuvée)

1. **Prix de référence proposé :** clôture native du perp BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT à la clôture M5 observée t. Le perp a une source vérifiée sur les 20 mois accessibles jusqu'à mai 2026. Ne pas mélanger spot, mark et perp dans le dénominateur ou le prix futur sans un protocole explicite.
2. **Horizon proposé :** H=6 bougies M5 exactement ; la maturité hypothétique est `t + 30 min`, jamais inférée d'un décalage de lignes en présence de trous.
3. **Variable mathématique proposée :** `r_H(t) = perp_close(t+H) / perp_close(t) - 1`. Calcul futur autorisé *seulement* pour un protocole supervisé ultérieur approuvé, jamais dans les features X(t).
4. **Seuil directionnel proposé :** `theta(t) = 0.5 * sigma_48(t) * sqrt(6)`, où `sigma_48(t)` est l'écart-type des 48 rendements logarithmiques à une bougie calculé exclusivement jusqu'à t sur une série perp continue. Cette formule est cohérente avec le concept du module historique `mlpipeline.labels`, **mais utilise un nouveau schéma de features V36 et n'implique pas de compatibilité automatique avec l'ancien entraîneur**.
5. **Classes proposées :** DOWN si `r_H < -theta`, UP si `r_H > theta`, NEUTRAL si `-theta <= r_H <= theta`. Aucun label lorsque le futur exact manque, les données ne sont pas finies, la continuité M5 est rompue, ou lorsque la maturité tombe à la frontière ou après la fin exclusive de la fenêtre train/validation/test.
6. **Purge stricte :** exiger `label_mature_at < window_end_exclusive` et conserver l'embargo V35 de 6 M5 aux débuts validation/test. Les 33 folds et leurs dates restent figés. Les indices et limites doivent être contrôlés avant de lire toute clôture future.
7. **Réserves :** aucune lecture de bougies juin 2026 (réserve interne) ni de juillet–septembre 2026 (holdout externe). Le funding est exclu faute d'horodatage démontré de disponibilité à t. Aucune hypothèse de rentabilité, de frais d'exécution ou de causalité d'une prédiction de trading n'est ici validée.

## Points bloquants AVANT tout label réel

- **Approuver la variable de cible et le prix de référence** (perp close plutôt que mark ou spot) en spécification versionnée distincte.
- Geler une règle de disponibilité ex ante pour chaque information utilisée à t ; aucune feature funding tant que l'heure de publication n'est pas démontrée.
- Définir une transformation des lignes natives continues en échantillons, avec contrôle de fuite temporelle et vérification qu'une étiquette n'est jamais une feature.
- Spécifier politique des doublons inter-folds, classes absentes, stabilité du seuil et interdiction de tuning sur les périodes aveugles.
- Concevoir d'abord des tests synthétiques de maturité, trous M5, frontières exclusives, inversion des seuils, NaN/inf, invariance de X(t) sous changement du futur.

## Statut

**TARGET_SPEC_PROPOSED_NOT_APPROVED — NO_LABELS_NO_TRAINING_NO_PNL_NO_ORDERS**.

V39 est un document de décision seulement. Aucun code de production ni programme de calcul de labels historiques n'est approuvé à ce stade.
