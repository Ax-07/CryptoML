# CryptoML — Export du modèle CPU et journal forward (après test final)

Cette phase vient **après** le déverrouillage définitif des 90 jours de test et d'ADA. Ne pas rejouer ces résultats pour sélectionner des paramètres.

## 1. Export reproductible sans contamination

Le programme \`mlpipeline.frozen_model\` reconstruit le **même** LightGBM CPU que lors du test final figé à partir des seules données pré-finales. Les étiquettes ADA et les 90 derniers jours ne servent ni à ajuster ni à évaluer le modèle pendant cet export. Le script bloque si les données, les bornes, le nombre optimal d'itérations, l'alpha choisi ou les scores de sélection ne correspondent pas au rapport final enregistré. Le modèle est stocké au format natif texte LightGBM, avec un SHA-256, une identité de modèle, les 30 features, leur ordre, les fréquences des classes et les bornes de formation.

Sous PowerShell, après tests :

\`\`\`powershell
cd E:\cryptoML-walkforward-test\backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.frozen_model \`
  --data-root "E:\cryptoML\data" \`
  --frozen-final-report "E:\cryptoML-walkforward-results\final_holdout_results.json" \`
  --outdir "E:\cryptoML-walkforward-results\frozen_cpu_lgbm_v1"
\`\`\`

Ce dossier est créé sans écraser un export existant. Si la reproduction échoue, **ne pas contourner les contrôles**, ne pas recalibrer alpha à partir du test final : étudier l'écart et recommencer seulement après justification, sans transformer le test final en jeu de développement.

## 2. Prédictions réellement postérieures au test final

Le programme \`mlpipeline.forward_signals\` **ne lit pas** les fichiers historiques d'étiquettes pour générer de faux signaux. Il charge un Parquet contenant les features **causales déjà calculées** pour une même clôture M5 fraîche sur chaque actif ; il rejette les timestamps passés, futurs, synchronisations incohérentes, prédictions enregistrées après l'heure d'entrée théorique, valeurs non finies et doublons.

Le Parquet doit contenir : \`symbol,bar_close_at\` (timezone UTC) et exactement les 30 \`FEATURE_COLUMNS\` dans \`mlpipeline.features\` (colonnes supplémentaires tolérées par le logger ; ne pas y mettre de futures cibles). Les features ne doivent dépendre que des bougies closes à \`bar_close_at\`. Elles doivent avoir été produites par un pipeline d'ingestion live/replay causal, **encore à raccorder**. Le logger ne récupère pas les données d'exchange et ne planifie pas de tâche de fond.

Exécution **seulement lorsqu'un Parquet authentiquement frais existe** :

\`\`\`powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.forward_signals \`
  --model-dir "E:\cryptoML-walkforward-results\frozen_cpu_lgbm_v1" \`
  --closed-features "E:\cryptoML-walkforward-results\current_closed_features.parquet" \`
  --log-dir "E:\cryptoML-walkforward-results\forward_signal_snapshots"
\`\`\`

Le programme capture lui-même l'heure UTC effective, exige une clôture M5 datant de moins de cinq minutes, et écrit un fichier JSON distinct par symbole et clôture, avec mode création exclusive. Ces fichiers JSON sont des **snapshots immuables par convention applicative**, pas un stockage certifié inviolable. L'agrégation et le rapprochement avec les outcomes pour \`paper_replay\` devront être réalisés séparément sans altérer les snapshots originaux.

## 3. Limites

- Ce programme ne gère ni exécution d'ordres, ni gestionnaire de tâches, ni serveur web ou websocket.
- La capture des données live causales, le stockage fiable du flux, les vrais prix d'exécution (et pas seulement les prix M5 moyens), la journalisation durable en présence de pannes et les contraintes opérationnelles restent à construire.
- Un modèle reconstruit à partir de données anciennes est figé sur l'état pré-juillet 2026. Ses performances sur des bougies fraîches post-septembre sont **inconnues**.
- Pas d'activation de trading avec de l'argent réel sur ces seuls résultats.


## 4. Adaptateur pour bougies M5 fraîchement clôturées

Le module \`mlpipeline.fresh_features\` prépare **les 30 features originales** à partir d'un fichier Parquet externe contenant au moins 240 bougies M5 **consécutives par crypto**. Il applique les mêmes fonctions historiques \`build_symbol_features\` et \`join_market_context\`, notamment pour le contexte BTC/ETH à la même clôture. Il refuse :
- une clôture non synchronisée sur les cinq cryptos ;
- une bougie en cours, absente, décalée de 5 minutes ou contenant des prix invalides ;
- une ingestion antérieure à la clôture ou postérieure à l'heure de capture ;
- un contexte technique insuffisant pour calculer toutes les features ;
- les colonnes d'étiquettes calculées à partir de données futures.

Format des colonnes : \`symbol,bar_open_at,bar_close_at,is_final,ingested_at,open,high,low,close,volume_base,volume_quote,trade_count,taker_buy_volume_base,taker_buy_volume_quote\`. Les trois horodatages sont explicitement UTC, conformément au schéma d'ingestion. Toutes les cryptos (\`BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, ADAUSDT\`) sont requises. Le fichier est un **lot récemment collecté**, pas les 24 mois déjà stockés.

Exécuter **seulement après mise en place de la collecte causale** et sur un vrai lot de données de marché :

\`\`\`powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.fresh_features \`
  --closed-candles "E:\cryptoML-walkforward-results\fresh_closed_m5.parquet" \`
  --output "E:\cryptoML-walkforward-results\current_closed_features.parquet"
\`\`\`

Puis appeler \`mlpipeline.forward_signals\` dans les cinq minutes suivant la clôture de la bougie. Les deux étapes doivent fonctionner **sur le même horodatage M5** ; le logger effectue sa propre vérification de fraîcheur, sans mode backfill.

**Limite non résolue** : la connexion à un flux d'ingestion live authentique, horodaté et durable, n'existe pas encore dans ces scripts. Ils ne doivent pas être utilisés pour transformer des fichiers historiques en prédictions soi-disant temps réel.
