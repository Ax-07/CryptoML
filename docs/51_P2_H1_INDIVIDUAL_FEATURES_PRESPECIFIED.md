# P2 — Pré-déclaration : H1, 30 indicateurs seuls et ajoutés à EMA9/26

**Date : 2026-10-19. Statut : recherche exploratoire sur checks P2 déjà inspectés ; aucune confirmation indépendante.**

## Question fixée avant les résultats

Le signal du **signe brut de la prochaine bougie M5 H1** provient-il d'un indicateur isolé ? Un indicateur original peut-il ajouter une information utile au LightGBM à seulement deux variables EMA9/26 ?

Comparer exactement **60 variantes** pour les **30 colonnes gelées de \`features_v1\`** :
- \`solo_<feature>\` : un LightGBM multiclasses H1 avec exactement cette colonne.
- \`ema2_plus_<feature>\` : LightGBM multiclasses H1 avec \`ema9_price_gap\`, \`ema9_ema26_gap\` et exactement cette colonne (total 3 variables).

Les deux EMA ne sont pas les EMA 12/48 présentes dans les 30 originales ; aucune recherche de période n'est autorisée. Variables temporelles sin et cos sont examinées **séparément**, bien qu'elles forment naturellement une paire : résultats non directement comparables à un encodeur complet. Les variables BTC/ETH sont présentes uniquement lorsqu'elles font partie du schéma figé, pas dans le modèle EMA2 de référence.

Les **modèles de référence LightGBM EMA2 et LightGBM 30** ne sont **pas réentraînés** : réutiliser leurs prédictions H1 produites dans \`p2_ema9_26_h1h6_full_predictions_v1.parquet\` et le rapport associé, en validant chaque clé (fold, symbol, close) et vérité H1. Les configurations de toutes les variantes sont gelées : LightGBM CPU 3 classes / 150 estimateurs maximum / seed 42 / early stopping 30 sur les mêmes labels à volatilité 0.5×sqrt(H), mêmes règles de départage UP/DOWN par majorité de train. Aucune optimisation de la complexité, de la direction ou du seuil selon les checks.

**Critère descriptif principal** : différence appariée en points de précision **du signe brut de H1 sur toutes les bougies non plates** entre \`ema2_plus_<feature>\` et **EMA2 inchangé**, avec moyenne sur les 36 folds et score global (les folds et bougies sont dépendants). Pour la variante \`solo_<feature>\`, référence identique EMA2 ; également comparer à LightGBM30. Les bougies futures plates sont comptées et exclues des deux côtés, dénominateurs rigoureusement égaux.

**Diagnostics secondaires** : signe H1 par actif (BTC/ETH/SOL/XRP) ; nombres de folds avec gain positif/négatif/nul ; désaccords gagnés/perdus sur mêmes lignes ; précision de la classe H1 forcée (neutre = erreur) sur toutes les lignes et sur le masque **gelé** \`baseline_shared_mask\` du modèle 30 ; logloss multiclasses brute H1, comparaison vs références déjà enregistrées ; précision du signe H1 sur \`ema2_autonomous_mask\` **gelé** (mêmes lignes pour tous les modèles, mais ce masque appartient au modèle EMA2).

**Aucun recalibrage/autosélection de seuil** par indicateur. Aucun classement présenté comme une preuve, aucune recherche a posteriori de "meilleure" variante, aucune promotion ou trading. 60 comparaisons ne sont pas indépendantes ; les 36 folds de développement se recouvrent. Toute confirmation nécessiterait des observations chronologiquement nouvelles et un protocole préenregistré.

## Géométrie et sécurité

Reprendre exactement les mêmes 36 folds de \`p2_feature_reduction_individual_remaining25_full_v1.json\` : train 120 jours, early 7 jours, tuning sans labels 3 jours, check 4 jours ; **purge H12 (60 minutes)** pour conserver les mêmes observations. Les labels H1 sont reconstruits par le code canonique **après** avoir tronqué à \`check_end\` propre à chaque fold, et leur vérité est recoupée avec le Parquet ancien. Aucun ADA, aucun test OOS de fold courant, aucun final90, aucun modèle P3. Données P0 en lecture seule, 4 actifs vus uniquement, manifeste et SHA vérifiés. Aucune modification du \`features_v1\` gelé.

**Sorties** : JSON rapport détaillé (60 variantes × folds × actifs) et CSV récapitulatif autonome trié en ordre **canonique** des 30 features, sans suppression des variantes défavorables. Les sorties sont créées avec des noms neufs hors production ; aucune écriture d'un modèle ou d'un fichier vers P0. Le smoke peut utiliser \`--max-folds 1 --max-features 1\` : 2 fits pour valider le pipeline, **pas** pour choisir une feature. Le full utilise les 30×2×36 = **2160 fits CPU**. Les six horizons du précédent benchmark restent inchangés ; cette expérience se limite volontairement à **H1**.

## Lecture attendue

Rapport sur chaque indicateur, **sans masquer les échecs** : performances \`solo\`, \`ema2_plus\`, EMA2, LightGBM30, différences appariées, plus/moins/égal par fold, comparaisons par crypto, logloss, et diagnostics à masque commun. Le classement visuel (si choisi par l'utilisateur) ne constitue pas une validation.
