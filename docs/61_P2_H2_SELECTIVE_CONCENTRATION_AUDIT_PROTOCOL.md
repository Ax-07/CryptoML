# P2 H2 sélectif — audit préfixé de concentration et de stabilité

**Objet :** diagnostic descriptif en lecture seule, sans fit, sans changement de seuil. Sources obligatoires : `p2_h2_delayed_selective_full_v1.json`, `p2_h2_delayed_selective_predictions_v1.parquet`, plan original des 36 folds P2. **Ne pas lire P0, P3, ADA, fold OOS ni final90.**

Modèle principal : `regression_30`, décision originale `+1` si rendement prédit strictement supérieur à +3 bp ; `-1` si strictement inférieur à −3 bp ; sinon `0` (abstention). Coût aller-retour principal **2 bp par trade** ; les abstentions coûtent zéro. Modèle de comparaison descriptive : `ridge_ema2`, sans modifier les paramètres. Les rendements du Parquet sont des *markouts non exécutables* entre l'OPEN de la bougie qui débute à t+5 min et son CLOSE à t+10 min. Aucune garantie de remplir ces prix, aucun ordre ni simulation de capital.

## Géométrie et invariant de rapprochement

Exiger le JSON complet (`protocol=p2_h2_delayed_return_selective_devchecks_v1`, 36 folds, empreinte P0 identique au plan, assertions de sécurité, coûts fixés), le Parquet sur les mêmes clés `(fold_id,symbol,bar_close_at)` avec 4 symboles connus seulement, timestamps UTC non dupliqués, `delayed_mature_at = bar_close_at + 10 min` et H12-purge sous `check_end`. Refuser toute dérive des signes, des probabilités prédites, des nombres de trades, des rendements des folds, du total ou du découpage par symbole. Le Parquet contient les issues historiques : elles sont utilisées **uniquement pour le diagnostic**, jamais pour recalibrer les décisions.

## Vues descriptives fixées à l'avance

1. Tous les folds : positions, couverture, gain brut et net par position / par opportunité, nombre de folds positifs/négatifs/sans trades, contribution en **bp additionnés sur les signaux** (non un PnL).
2. **Leave-fold-1-out** et **leave-one-fold-out pour chacun des 36 folds**, sans en retenir un pour choisir un modèle. Comptage et somme de contributions, gain net par trade de la population restante. Distinguer contribution du fold 1, sa part des positions et sa part de la somme positive.
3. Répartition par actif, sens LONG/SHORT, UTC jour et fold. Deux diagnostics temporels : jours cumulant le plus de résultats positifs, et séquences consécutives de signaux espacés de 5 min **dans le même actif et fold**. Les journées ne sont pas des échantillons indépendants.
4. Queue des gains/pertes nettes à 2 bp : quantiles prédéfinis 0, 1, 5, 25, 50, 75, 95, 99, 100 % ; 5 meilleurs/pires trades et leurs identifiants ; part des 1/5/10 meilleurs trades dans la contribution totale ; résultats après suppression purement descriptive des top 1/5/10 gains nets et pertes nettes, sans refit. Aucun seuil de trading n'est modifié.
5. Répétitions des mêmes `(symbol, bar_close_at)` **entre folds** et leur contribution. Les folds se chevauchent et les rendements du même marché peuvent compter plusieurs fois ; une vue diagnostique dédupliquée déterministe, gardant le plus petit `fold_id`, est rapportée comme sensibilité, **pas comme portefeuille négociable**. Les markouts côte à côte de cryptos ne sont pas un capital commun.

Les sorties sont JSON structuré et CSV récapitulatif. Aucun fichier existant n'est écrasé. **Conclusion interdite :** preuve d'edge, rentabilité, validation OOS ou de déploiement. Ce protocole est strictement post hoc sur des checks réutilisés.
