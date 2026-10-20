# CryptoML — Observateur public Binance M5 (CPU, lecture seule)

Le projet dispose d'un nouveau point d'entrée \`mlpipeline.live_observer\` qui réutilise \`ingestion.sources.BinanceClient.fetch_rest\` et \`ingestion.schema.normalize_klines\` déjà présents dans CryptoML, suivis du calcul des 30 features et, optionnellement, de l'inférence LightGBM sur son bundle figé.

**Première vérification recommandée :** aucune écriture de prédiction et aucun modèle requis.

\`\`\`powershell
cd E:\cryptoML-walkforward-test\backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.live_observer --check-only
\`\`\`

Cette opération appelle uniquement l'API publique \`GET /fapi/v1/klines\` du marché USD-M Binance pour BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT et ADAUSDT, chacune sur au plus 300 bougies M5, puis valide les dernières clôtures. Elle **ne passe aucun ordre**, n'utilise aucune clé d'API et ne modifie pas \`E:\cryptoML\data\`.

**Mode snapshot à la demande :** seulement une fois \`--check-only\` validé, et lorsque le dossier de modèle figé est disponible.

\`\`\`powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.live_observer \`
  --model-dir "E:\cryptoML-walkforward-results\frozen_cpu_lgbm_v1" \`
  --log-dir "E:\cryptoML-walkforward-results\forward_signal_snapshots"
\`\`\`

Le mode snapshot est intentionnellement **manuel et one-shot** (pas de boucle, de scheduler, de daemon, ni d'ordres). Il écrit une fois par couple symbole/clôture et refuse les doublons. Il peut échouer si le dernier kline officiel n'est pas encore publié, si la collecte traverse une frontière M5, si la fenêtre présente un trou ou si l'inférence n'a pas fini avant la clôture M5 suivante. Cela est normal : ne pas rétro-remplir les signaux.

**Aucune performance de trading** n'est présumée. Il reste à écrire le règlement causal des trades virtuels dans \`paper_replay\` à partir de snapshots réels archivés, et à traiter la fiabilité d'une collecte en continu, qui n'est pas mise en place dans cette étape.

### Qualité et sécurité

- Tous les appels Binance sont **GET publics** via le client existant ; aucune action en compte.
- \`lookback_bars\` borné entre 240 et 500, sans pagination infinie.
- Un seul horodatage UTC de clôture M5 doit être disponible simultanément pour les 5 cryptos.
- La continuité de la fenêtre et l'absence d'observations futures sont contrôlées.
- Pas d'écriture dans le stockage historique P0 ni dans les résultats du protocole final.
- Si la requête HTTP dure trop longtemps ou si Binance renvoie des bougies manquantes, le programme échoue sans créer de fausses observations.

API de référence : documentation Binance USD-M, <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data>.
