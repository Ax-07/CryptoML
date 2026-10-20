# CryptoML — Paper replay CPU, protocole après holdout

## État et portée

Le test final du 2 juillet au 30 septembre 2026 **a déjà été consulté**. Les paramètres du modèle, ses classes, et la grille d'ajustement ne doivent plus être modifiés à partir de ces scores. Résultats figés :

| Ensemble | Modèle | Macro-F1 | Balanced accuracy | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| BTC/ETH/SOL/XRP | LightGBM | 0.3869 | 0.3937 | 1.0490 | 0.6316 |
| BTC/ETH/SOL/XRP | Logistique | 0.3854 | 0.3944 | 1.0516 | 0.6328 |
| BTC/ETH/SOL/XRP | Prior | 0.2093 | 0.3333 | 1.0658 | 0.6437 |
| ADA | LightGBM | 0.3797 | 0.3802 | 1.0626 | 0.6413 |
| ADA | Logistique | 0.3828 | 0.3839 | 1.0630 | 0.6410 |
| ADA | Prior | 0.2058 | 0.3333 | 1.0712 | 0.6475 |

Aucun de ces scores ne représente un PnL ou une validation de trading.

## Design du replay

Le code \`backend/mlpipeline/paper_replay.py\` est **offline seulement**. Il n'envoie ni ordre, ni requête à un exchange. Il est séparé du pipeline d'entraînement. Le replay requiert un historique authentique de signaux prédits et **journalisés au moment du signal**, des résultats observés seulement une fois la période de détention écoulée, ainsi que le rapport final déjà gelé.

1. **Planification isolée des résultats** : \`plan_positions\` n'accepte ni \`future_return\` ni \`target\` ni prix futur dans le journal des signaux. Tout signal doit être horodaté après la clôture M5 et avant l'entrée supposée (5 min après) et être postérieur à la fin du holdout le 30 septembre 2026 à 23:35 UTC. Un seul ID de modèle et une même distribution de classes d'entraînement, figés, par exécution.
2. **Décision** : probabilités LightGBM/logistique brutes, priors d'entraînement immuables et alpha du rapport final (choisi sur validation pré-finale). Un seul signal LONG/SHORT par crypto ouverte et capacité maximale de positions simultanées. Les signaux NEUTRAL et les marges trop faibles conduisent à s'abstenir.
3. **Règlement ultérieur** : jointure stricte sur \`symbol\` et \`entry_at\`. Entrée théorique au cours médian à \`bar_close_at+5 min\` ; sortie 30 min après l'entrée (6 bougies M5). Le prix de sortie doit avoir été enregistré au moins à l'heure de sortie. Frais et slippage appliqués **sur les deux jambes** ; notionnel de 100 USDT par trade par défaut, sans effet de levier modélisé ni réinvestissement.
4. **Reporting** : fichier \`report.json\`, liste \`paper_trades.parquet\` séparée ; nombre de trades, net PnL notionnel, taux de trades gagnants et drawdown en PnL cumulé *à la clôture* uniquement. Les résultats ne doivent jamais être utilisés pour ajuster la stratégie figée sous couvert d'évaluation prospective.

**Hypothèses et limites** : l'entrée au cours médian futur + glissement forfaitaire n'est pas une simulation réaliste d'ordres à carnet ; slippage, bid/ask, latence, funding, liquidation, impacts de marché, bornes de capital et mark-to-market intratrade ne sont pas reproduits. Les trades de 30 min chevauchent les labels ; aucune indépendance statistique implicite. Le \`net_pnl_usdt\` ne représente pas une courbe de rendement d'un capital réellement immobilisé.

## Format d'entrées

\`signals.csv\` ou Parquet : \`symbol,bar_close_at,recorded_at,model_trained_until,model_id,model_family,p_down,p_neutral,p_up,prior_down,prior_neutral,prior_up\`.

Chaque ligne comporte une clôture UTC, un horodatage d'enregistrement UTC réel et un \`model_trained_until\` qui n'est pas postérieur au début du holdout. \`model_family\` est \`lightgbm\` ou \`logistic\`. Les probabilités et priors suivent **DOWN / NEUTRAL / UP** ; les priors d'entraînement doivent être strictement positifs et constants sur tout le fichier. Le signal contient des probabilités réellement produites par un modèle préalablement gelé : ce module ne construit ni entraîne ni exporte ce modèle.

\`outcomes.csv\` ou Parquet : \`symbol,entry_at,exit_at,entry_mid,exit_mid,observed_at\`. Ce fichier est établi *après* la clôture de sortie de chaque trade. L'horodatage \`observed_at\` doit être postérieur à \`exit_at\`. Il faut les prix pour **tous** les trades retenus. Ne pas remplir artificiellement les prix manquants.

Exécution **uniquement lorsque les vrais fichiers de signaux et d'outcomes existent** :

\`\`\`powershell
cd E:\cryptoML-walkforward-test\backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.paper_replay \`
  --signals "E:\cryptoML-walkforward-results\forward_signals.csv" \`
  --outcomes "E:\cryptoML-walkforward-results\forward_outcomes.csv" \`
  --frozen-final-report "E:\cryptoML-walkforward-results\final_holdout_results.json" \`
  --model-family lightgbm \`
  --fee-bps 5 \`
  --slippage-bps 5 \`
  --notional-usdt 100 \`
  --max-positions 4 \`
  --min-margin 0 \`
  --output-dir "E:\cryptoML-walkforward-results\paper_replay_forward_v1"
\`\`\`

**Interdit** : ne pas renommer les anciennes prédictions OOS en \`forward_signals.csv\`. Ne pas récupérer des rendements des fichiers étiquetés historiques pour fabriquer des signaux datés rétrospectivement. Une simulation purement synthétique via les tests \`pytest\` ne vaut pas validation forward.

## Avant utilisation réelle

- Ajouter au projet un mécanisme de **sauvegarde de modèles gelés** : les scripts d'évaluation finale écrivent actuellement des métriques, mais ne sérialisent pas leurs modèles ; il n'existe donc pas encore de producteur de \`forward_signals\` prêt à l'emploi.
- Prévoir une génération de signaux sur bougies fraîchement clôturées, conservation des hashes et ID du modèle, heures d'observation, journalisation non modifiable et vérification de la disponibilité des données de marché.
- Effectuer une revue du code, des tests unitaires et du protocole avant d'exploiter des résultats économiques.
- Ne pas fusionner cette PR dans \`main\` avant rapprochement des modifications locales P1.1 / \`lab.py\`.
