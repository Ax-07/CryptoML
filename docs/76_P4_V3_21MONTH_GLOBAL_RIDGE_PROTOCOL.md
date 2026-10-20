# P4 V3 — Protocole prédictif 21 mois, préspecifié (avant étiquettes V3)

**Statut : FROZEN DESIGN / aucun entraînement V3 exécuté.**
Date : 2026-10-20. Parent : `research/trend-pullback-long-history-v3`.
Ne pas modifier rétroactivement la V1 ou la V2 et ne pas prétendre à une validation indépendante sur les mêmes périodes explorées.

## Données observées avant gel du protocole

Le recensement **sans étiquettes de performance** sur les 21 mois de développement a été exécuté avec succès sur le poste de l'opérateur :
- 734 972 bougies M5 clôturées, 21 583 signaux V1 après cooldown, 0 non classifié ;
- Octobre 2024 à novembre 2025 : 14 769 signaux (14 mois supplémentaires) ;
- Décembre 2025 à juin 2026 : 6 814 signaux, **identiques crypto par crypto** au recensement initial sur 7 mois ;
- BTC : 5 480 (3 763 / 1 717), ETH : 5 432 (3 703 / 1 729), SOL : 5 368 (3 652 / 1 716), XRP : 5 303 (3 651 / 1 652).
Les chiffres sont les sorties de console transmises par l'opérateur, non un audit local des Parquets dans l'environnement du rédacteur.

Cet élargissement multiplie les échantillons **de signaux non labellisés** par ~3.17. Le volume de B confirmés et leur rentabilité ancienne restent inconnus à cette étape : **ne pas les inventer**.

## Hypothèses fixes, une seule analyse exploratoire

- **Primaire** : modèle global Ridge 19 features (9 V1 confirmation + 10 V2 régime) vs Ridge 9 features, entraînés sur **la même profondeur de 18 mois**, mêmes événements, même coût, même information temporelle et même calibration. Effet incrémental des features de régime.
- **Secondaire** : Ridge 9 features entraîné avec **18 mois** vs Ridge 9 features historique (V1 entraîné avec 4 mois), pour étudier le seul effet d'élargir l'historique. Cette comparaison peut différer légèrement si un indicateur a un historique supplémentaire ; expliciter les identités de signaux avant d'interpréter.
- **Benchmarks nécessaires** : baseline A tous les pullbacks mûrs, baseline B confirmation mécanique t+1, référence zéro trade ; sélection aléatoire analytique d'**égale couverture par crypto et mois** pour chaque modèle (et non uniquement un rendement par opportunité).
- Aucune autre architecture, aucune grille d'hyperparamètres, aucune recherche de sous-groupes/buckets. Tout résultat négatif doit être conservé.

## Caractéristiques et horloge causales

- Événement V1 : premier toucher EMA9 au close(t), cooldown V1 inchangé.
- B : confirmé au close(t+1) avec les trois règles déjà fixes ; hypothèse d'entrée à l'OPEN(t+2).
- Features V1 (9) : `pullback_ml.FEATURES` calculées après close(t+1) ; features V2 (10) : `pullback_regime_v2.FEATURE_COLUMNS` au close(t), puis jointure **univoque** par (symbol, signal_close_at). Les features ne peuvent dépendre ni de l'OPEN(t+2) ni des barrières/exits.
- Label unique pour fit : `net_5bps` théorique de `research_outcomes`, même stop 1 ATR, cible 1.5 ATR, six bougies M5, cas intrabar ambigus = stop. **Les labels ne sont jamais des features**.
- Modèles `StandardScaler + Ridge(alpha=10)`, `clip(y_train, −100, +100 bp)` **sur train uniquement**.
- Seuil par modèle fixe : `predicted_net_5bps > max(0, percentile_80(scores sur B avril 2026))`. N'utiliser aucune étiquette d'avril pour calibrer. Abstention si aucun score admissible.
- Frais aller-retour théoriques 2, 5, 10 bp : rapporter chaque sensibilité après sélection sans rechoisir les seuils.
- Entrée au prix OPEN(t+2) = **hypothèse de recherche**, non exécution réalisable garantie au même instant que la décision close(t+1).

## Temps, séparations et maturité

Les partitions sont figées dès maintenant :
- `train` = **2024-10-01 inclus → 2026-04-01 exclus** : 18 mois ; tous les labels doivent avoir une maturité strictement avant 2026-04-01.
- `calibration` = **2026-04-01 → 2026-05-01** exclus : uniquement les scores B ; pas les rendements pour fixer la politique.
- `diagnostic_1` = **2026-05-01 → 2026-06-01** exclus, et `diagnostic_2` = **2026-06-01 → 2026-07-01** exclus ; un événement n'est inclus que si son `mature_at` est strictement avant la fin de sa propre partition.
- La purge couvre la maturité complète des labels (au moins 35 minutes après close(t)). Aucun événement en lisière de frontière ne traverse une partition.
- Les mêmes opportunités A et B doivent composer les deux modèles dans chaque groupe actif×mois, sauf cas rapporté et justifié de feature manquante. Ne pas laisser un modèle choisir secrètement un sous-univers plus facile.
- Tous les résultats historiques sont **exploratoires** : mai-juin ont déjà été consultés, et les 14 mois plus anciens sont ajoutés après l'inspection de résultats V1. Ni un score positif ni une bonne statistique cross-month ne prouvera un edge exploitable.

## Contrôles nécessaires avant exécution

1. Chargeur P0 `load_long_development_history` vérifie exactement 84 manifestes (4 actifs × 21 mois), intégrité SHA-256, bougies finalisées et coupe **avant 2026-07-01**. Ne lire aucun Parquet ADA ni juillet-septembre 2026, ni résultat P3.
2. `annotate_confirmation` et `pullback_ml.labeled_events` V1 sont **volontairement bornés par START=2025-12-01** : il faut une **implémentation V3 distincte**, sans changer la version V1. Reprendre fidèlement la règle B et ses horloges, puis vérifier synthétiquement l'identité sur le segment commun.
3. Horodatages UTC uniques, continuité t, t+1, t+2 et six bougies d'issue ; pas d'entrée sur un trou M5, pas de forward fill. Vérifier le décompte exact 21 583 signaux avant filtrage de maturité.
4. Invariance causale : modifier les bougies futures après close(t+1) ne doit pas modifier les vecteurs de features des décisions antérieures.
5. Tests synthétiques de purge aux frontières, de conservation de volume, de seuil/bornes et de stricte séparation features/labels.
6. Comparaison à égalité de **trades par crypto×mois** (espérance sous tirage aléatoire : moyenne B du groupe multipliée par le nombre choisi). Reporter performance **net/trade ET net/opportunité**. Un filtre qui fait simplement moins de trades n'a pas prouvé de sélection.
7. Référence 0 trade = 0 net toujours présente ; pas de promotion au live sur le développement réutilisé.
8. Sorties nouvelles, immuables, contenant fingerprint P0, code/protocole, coûts, splits, effectifs et avertissement explicite `NOT_INDEPENDENT_OOS`.

## Décision

Exécuter **une seule analyse V3 figée** une fois les nouveaux tests réussis. Si les modèles ne sont pas meilleurs que l'espérance du hasard à même allocation et que zéro net après 5 bp, statut `NO_ELIGIBLE_CANDIDATE`, sans optimisation ultérieure sur les diagnostics consultés.

Pour obtenir une preuve confirmatoire : geler code, features, modèle, seuil et hypothèses d'exécution **avant des prédictions futures réellement inédites**, conserver leurs traces horodatées et évaluer ensuite les résultats coûts/slippage/fills effectivement obtenus.
