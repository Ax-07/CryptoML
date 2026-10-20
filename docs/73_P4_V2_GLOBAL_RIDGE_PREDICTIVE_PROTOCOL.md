# P4 V2 — Protocole prédictif global, gelé avant l'ouverture des résultats V2

**Statut : SPECIFICATION ONLY — aucun entraînement V2 autorisé par ce document seul.**
Date : 2026-10-20. Branche : `research/trend-pullback-regime-v2`.
Conserver les résultats P4 V1 déjà explorés comme contexte de recherche et non comme résultat de validation indépendante.

## Pourquoi un seul modèle global ?

Recensement descriptif causal vérifié (6 814 événements, 0 non classifié) :

| Force EMA9−EMA26 / ATR | Toucher EMA9 [0,0.5) ATR | Repli [0.5,1) ATR | Repli ≥1 ATR |
| --- | ---: | ---: | ---: |
| Faible (<0.5 ATR) | 1 936 | 100 | 0 |
| Moyenne ([0.5,1.5) ATR) | 3 521 | 778 | 43 |
| Forte (≥1.5 ATR) | 383 | 49 | 4 |

La classe `no_undercut` est toujours nulle par définition V1. Un modèle différent pour chaque combinaison serait peu défendable, vu les effectifs et les découpages temporels. **Ne sélectionner aucun bucket gagnant a posteriori.**

## Hypothèse explicite V2

Ajouter les dix features causales de régime V2 définies dans `pullback_regime_v2.FEATURE_COLUMNS` aux neuf features fixes de la confirmation V1 `pullback_ml.FEATURES` permettrait-il d'améliorer **la qualité par trade sélectionné**, et pas seulement de diminuer la couverture ?

L'unité statistique reste **un événement V1 retenu ayant la confirmation mécanique B au close(t+1)**. Les features V2 sont figées au close(t), celles de V1 au close(t+1). La décision précède l'OPEN hypothétique(t+2), aucune feature ne dépend des prix d'entrée/sortie futurs ou du résultat de trade.

Modèles : **V1 Ridge 9 features vs V2 Ridge 19 features**, tous deux StandardScaler + Ridge(alpha=10), cible = résultat net théorique à 5 bp par trade, *clip* d'entraînement à ±100 bp, sans recherche d'hyperparamètres. Même univers, même purge, même hypothèse de coûts et mêmes signaux disponibles. Ne pas comparer une décision au close(t) à une autre décision au close(t+1).

## Partitions fixes et non indépendantes

- Train : 2025-12-01 inclus → 2026-04-01 exclus, événements B dont le résultat est mûr avant la fin.
- Calibration **des scores uniquement** : avril 2026, résultats mûrs avant 2026-05-01. Seuil de décision fixé `score > max(0, percentile_80(scores_avril))`. Ne pas utiliser les rendements d'avril pour choisir seuil, variables ou algorithme.
- Diagnostics : mai et juin 2026, résultats mûrs avant la fin de chaque mois ; ces périodes ont déjà été examinées en V1. **Elles ne sont pas OOS indépendant.**
- Au moins 35 minutes de purge effective à chaque frontière, par `mature_at < end` ; pas de label qui traverse une frontière.
- Chargement exclusif des partitions P0 BTC/ETH/SOL/XRP de décembre 2025 à juin 2026, avec vérification SHA-256 via `load_clean_historic`. **Aucune ouverture d'ADA, juillet 2026 ou de données plus récentes.**

## Comparaisons obligatoires sur les mêmes opportunités

1. `A` : tous les pullbacks mûrs, `B` : ceux confirmés t+1, `C1` : Ridge V1 neuf variables, `C2` : Ridge V2 dix-neuf variables, `D` : aucune opération (= zéro).
2. Coûts totaux supposés aller-retour : 2, 5, 10 bp. Les rendements restent des **markouts théoriques**, pas une simulation de portefeuille. Déclarer trades, couverture parmi toutes les opportunités A, net/trade et net/opportunité.
3. Pour C1 et C2, comparer leur net/trade à **l'espérance analytique d'un tirage aléatoire parmi B**, avec exactement le même nombre de trades dans **chaque groupe actif × mois**. `E[net_random] = nb_selected * mean(net_B)`. Ce n'est ni une permutation réalisée, ni une p-value. Ne pas prétendre à une significativité.
4. Décrire séparément les huit groupes BTC/ETH/SOL/XRP × mai/juin. Ne pas promouvoir sur le seul global positif ou sur un mois/actif favorable.
5. Aucun modèle par bucket, aucun filtrage du petit groupe profond, aucune optimisation de seuil après examen des diagnostics. Documenter la sortie même si `C2` ne prend aucun trade.

## Critère de recherche et décision

Une moyenne positive à 5 bp, supérieure à zéro et à la sélection aléatoire de même couverture, serait **un signal exploratoire à approfondir**, pas une preuve d'avantage. Si V2 perd ou est instable, documenter sans réajuster. Il n'existe **aucun critère de promotion au live à partir des mêmes mois historiques**.

La validation confirmatoire exige une politique définitivement gelée et des prédictions **vraiment futures**, horodatées extérieurement avant toute observation des résultats, puis une évaluation économique réaliste (spread, slippage, coût effectif, exécution, exposition, drawdowns).

## Barrières à tester avant tout entraînement

- Pas de future leakage : modifier toutes les bougies après la clôture t+1 sans changer les features de sélection à t+1.
- Test synthétique de maturité/purge aux frontières mensuelles.
- Cohérence exacte des événements et entrées V1/B avec les features V2 jointes, sans doublon ni lookahead.
- Comptage des signaux V2 inchangé : 6 814 avant exclusion des événements sans issue mûre.
- Même dénominateur d'opportunités pour tous les candidats ; référence zéro explicite.
- Pas d'écrasement de rapports antérieurs ; empreinte de jeu de données dans la sortie.

**Prochaine étape de code : implémenter ce protocole sans lancer de recherche d'hyperparamètres, tester synthétiquement sur Windows, puis exécuter une seule fois les diagnostics historiques — uniquement après validation des tests.**
