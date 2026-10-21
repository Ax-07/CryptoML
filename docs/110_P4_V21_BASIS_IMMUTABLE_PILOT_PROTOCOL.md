# P4 V21 — Protocole gelé : ingestion pilote BTCUSDT juin 2025

**Gel : 21 octobre 2026, avant ingestion V21.** L'utilisateur a confirmé que les tests V20 passent et que l'archive Binance Futures `BTCUSDT-fundingRate-2025-06.zip` comporte **90 événements** avec `calc_time`, `funding_interval_hours` et `last_funding_rate`. V19 : trois archives de prix (spot, USD-M perpetual, mark) avec 8640 bougies M5 chacune et checksum vérifié. Leurs horodatages communs doivent être validés séparément dans le rapport JSON V19.

**Pas de stratégie, pas de PnL, pas de backtest, pas de ML, pas d'ordre.** Les archives spot ne sont pas une preuve d'exécution au bid/ask ; mark price n'est pas spot exécutable. Taux de funding réalisés **connus à la date d'événement**, non d'avance.

## Objet

Créer une nouvelle partition **hors du P0** contenant :
- quatre ZIP mensuels bruts BTCUSDT 2025-06 (`spot`, `perp`, `mark`, `funding`) et leur fichier CHECKSUM officiel, tous SHA-256 vérifiés ;
- `markets_5m.parquet` : **8 640 M5 communes**, `bar_open_at` en UTC, `bar_close_at = open+5m` exclusif, OHLC de spot, perp et mark, volumes spot/perp. **Pas de jointure ou forward-fill de funding dans M5** ;
- `funding_events.parquet` : **90 taux réalisés**, temps `calc_time` UTC, taux en fraction décimale, intervalle en heures tel que l'archive l'énonce ;
- `manifest.json` : SHA-256 de chaque ZIP, chaque CHECKSUM, chaque Parquet produit, plus SHA-256 des rapports V19 et V20 et du Parquet P0 (BTCUSDT juin 2025) utilisé pour une comparaison OHLC.

## Contraintes bloquantes

1. Lire les rapports locaux V19 et V20 et vérifier la chaîne SHA : `V20.v19_report_sha256` égale réellement les octets du V19 JSON, `V20.funding.status=FUNDING_SCHEMA_AND_CHECKSUM_OK`, `V19 spot/perp/mark=CHECKSUM_AND_SCHEMA_OK`, les archives V19/V20 déjà checksum-verified, et **8640 timestamps communs** dans V19. Les rapports ne remplacent pas un nouveau contrôle SHA des ZIP.
2. Télécharger sur action explicite **seulement quatre ZIP + quatre CHECKSUM** de BTCUSDT juin 2025, avec des limites d'octets et un timeout. Ne pas faire de scan en masse.
3. Recalculer les SHA-256 après téléchargement, rejeter le moindre écart, même si le CDN a remplacé un ZIP auparavant accepté. Garder le vrai ZIP comme preuve immuable.
4. CSV unique sans extraction du ZIP sur le disque, limites de décompression, protection path traversal. Gérer timestamp **microsecondes spot** (depuis janvier 2025), millis futurs lorsque observés, sans inférer à partir de la longueur d'une chaîne mélangeant unités ; rejeter toute discordance et les prix non finis.
5. Chaque marché doit contenir exactement 8640 M5, horodatées régulièrement du 1er juin 00:00 UTC au 30 juin 23:55 UTC. **Intersection exacte 8640**. Contrôler géométrie OHLC, prix strictement positifs, volumes non négatifs et `close_time` fournisseur égal à `open+5m−1` unité de timestamp (quand présent).
6. Comparer OHLC perp de l'archive originale au BTCUSDT 2025-06 du P0 existant **sans modifier le P0** : manifest qualité `OK`, SHA et row count vérifiés, mêmes M5 et prix numériques à tolérance d'arrondi bornée, aucun trou.
7. Funding : exclusivement le schéma V20 déjà constaté (calc_time, funding_interval_hours, last_funding_rate), taux décimal fini, plausible, timestamp croissant unique à l'intérieur du mois, intervalle 1–24 heures, **90 événements** comme annoncé ; ne pas fabriquer les événements manquants ni présumer trois par jour pour d'autres mois.
8. Stockage dans un dossier destination **nouveau**, en staging puis promotion finale ; ne jamais écraser un dossier existant. Ne pas ingérer en P0. Aucune suppression silencieuse de données non conformes. Le pilote ne permet pas d'affirmer que 21 mois × 4 cryptos sont disponibles ou complets.

## Ensuite

Sur validation V21, planifier une ingestion incrémentale avec inventaire 4×21×4, checksum de chaque source, schémas versionnés et gel du dataset avant de définir B01 (carry delta-neutre long spot/short perp) et B02 (basis). Le backtest devra gérer cashflow funding à sa **date réelle**, coûts deux jambes, prix exécutables et spread, slippage, risque de marge et liquidation, disponibilité d'emprunt, rebalancing, contraintes de capital, benchmarks et **NO_TRADE**. Le passé 2024–2026 déjà inspecté reste exploratoire ; validation indépendante à construire ultérieurement.

Sources : https://github.com/binance/binance-public-data (note timestamps spot µs en 2025) ; https://www.binance.com/en/support/faq/detail/360033525031 (paiements funding, fréquence variable).
