# P2 — Protocole figé : apprentissage direct du signe binaire H1

**Statut : EXPLORATOIRE, anciens checks développement déjà inspectés, pas d'indépendance OOS.** Aucun test 72 h nécessaire à cette étape.

## Hypothèse

Nos anciens modèles LightGBM H1 étaient entraînés à trois classes (DOWN/NEUTRAL/UP selon volatilité), mais jugés sur la couleur brute de la prochaine bougie M5. Cette expérience teste **le même objectif à l'entraînement et à l'évaluation**, en fixant quatre modèles :

1. \`binary_ema2\` : LightGBM binaire, EMA9-price gap et EMA9-EMA26 gap.
2. \`binary_ema2_close\` : LightGBM binaire, les mêmes deux EMA + \`close_location\`.
3. \`binary_30\` : LightGBM binaire, schéma canonique complet de 30 features.
4. \`logistic_ema2\` : StandardScaler entraîné sur train puis LogisticRegression, seulement deux EMA, hyperparamètres déterministes fixes.

**Cible fixe :** \`y=1\` si \`close(t+5min) > close(t)\`, \`y=0\` si inférieur. **Retours strictement nuls : exclus du train, de l'early stopping et de la métrique binaire primaire ; inclus dans le compteur de flat à chaque fold.** Aucun seuil de volatilité pour la cible binaire. Le modèle estime \`P(UP | mouvement non nul, X)\` à l'entraînement ; à l'inférence, le mouvement non nul n'est pas connu, et une prédiction UP/DOWN est néanmoins émise à chaque close. Pas d'oracle de sélection à l'inférence. \`p_up>0.5\` = UP ; \`<=0.5\` = DOWN (tie fixe DOWN).

Hyperparamètres figés : \`_cpu_model(binary=True,n_estimators=150,seed=42)\` (LightGBM CPU, arrêt anticipé 30 sur early **nonflat** uniquement, métrique binary_logloss) ; pour la logistique \`StandardScaler + LogisticRegression(C=1.0,max_iter=1000,solver="lbfgs",random_state=42)\`, entraînée sur train nonflat seulement, aucun early stopping.

## Comparateurs & métriques

Les deux modèles LightGBM H1 trois classes \`lightgbm_ema2\`, \`lightgbm_30\` sont **réutilisés à l'identique** depuis l'export \`p2_ema9_26_h1h6_full_predictions_v1.parquet\`, avec contrôle du rapport et des clés, jamais réentraînés. Références supplémentaires ex ante : toujours DOWN, reversion du rendement M5 \`ret_1\` connu au close t, tie DOWN.

**Principal descriptif :** précision de la direction brute H1 sur **toutes les bougies non plates** de chaque check. Comparaisons appariées à EMA2 trois classes et LightGBM30 trois classes : les mêmes observations pour tous. Agrégation pondérée par nombre d'observations **et** moyenne non pondérée par fold, folds gagnés/perdus/égaux. Par actif BTC/ETH/SOL/XRP.

**Secondaires non sélectifs :** Brier et logloss binaires sur lignes non plates, AUC quand les deux classes présentes, 4 tranches de marge \`|p_up - 0.5|\` FIXES (<0.01, 0.01–0.025, 0.025–0.05, >=0.05), et vues facultatives utilisant les anciens masques gelés \`baseline_shared_mask\`, \`ema2_autonomous_mask\` pour comparaison appariée. Aucune optimisation d'un seuil ni sélection automatique sur les checks, aucune prétention de calibration sans correction du filtrage flat.

## Géométrie et frontières

Les **36 folds P2 originaux** train 120 j, early 7 j, tuning 3 j, check 4 j, H12 60 min de purge de maturité avant tout split, sont réutilisés strictement. Construction des features/labels avec les fonctions canoniques **après clip de chaque fold à check_end**, comme l'étude H1–H6. Ni les 3 jours de tuning, ni les 4 jours de check ne sont employés pour ajuster ou recalibrer un modèle. Aucun accès aux fold OOS, au holdout final90 ou à ADA. Lecture seule de P0 sur le checkout Windows isolé, pas de P3.

Le livrable comprend JSON détaillé, CSV récapitulatif à ordre fixe et Parquet des prédictions H1 des nouveaux modèles + références et données de vérification (uniquement sur anciens checks inspectés). Smoke limité au premier fold (\`--max-folds 1\`) à titre de test technique uniquement. Full 36 folds = 108 entraînements LightGBM binaires + 36 régressions logistiques. Pas de suivi de 72 h, de service ni d'ordres.
