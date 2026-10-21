# CryptoML V19 — Audit d'intégrité des archives spot/perp/funding

**21 octobre 2026. Pas de backtest, pas de calcul de basis, pas d'ordre.**

V18 a confirmé 12 URL BTCUSDT par HEAD 200 (4 familles × octobre 2024, juin 2025, juin 2026), sans vérifier leurs octets. Un GET REST de funding a rendu 7 événements du 1er au 3 juin 2025 ; c'est un échantillon, non une garantie de couverture des 21 mois.

## Échantillonnage gelé et borné

Télécharger sur demande explicite uniquement **quatre archives ZIP BTCUSDT juin 2025** : `spot/monthly/klines`, `futures/um/monthly/klines`, `futures/um/monthly/markPriceKlines`, `futures/um/monthly/fundingRate`, plus leur `.CHECKSUM` respectif. Aucun scan massif, aucune nouvelle base P0, aucun trading. Les sorties sont une **synthèse JSON** et aucun OHLC/financement n'est utilisé pour une décision économique.

## Tests fail-closed

- Requête HTTPS avec timeout, limite d'octets ZIP par fichier et vérification d'un SHA256 hexadécimal présent dans le CHECKSUM officiel ; aucune validation par seul HTTP 200.
- Un membre CSV unique dans chaque ZIP, aucun chemin relatif ou absolu dangereux ni décompression sur disque. Protection taille décompressée et compression ratio.
- Identifier en-têtes facultatifs et colonnes réelles : spot et perp M5 au moins open time, OHLC, volumes ; mark au moins open time/OHLC ; funding avec timestamp et taux parsables. **Pas d'hypothèse de schema fixe des archives de funding non inspectées** : l'outil peut retourner `SCHEMA_REVIEW_REQUIRED` sans déclarer une couverture valide.
- Détecter timestamps ms ou µs explicitement (et s'ils apparaissaient, s) ; rejeter unité incohérente, tranches non monotones, doublons, timestamps de mois incorrect, M5 désalignés.
- Les timestamps sont des *opens* d'archives; les bougies ne deviennent disponibles qu'après leur clôture. Ne pas interpréter des close times Binance inclusifs comme des closes de notre convention exclusive.
- Comparer seulement **les horodatages d'ouverture** spot/perp/mark dans cet échantillon. Signaler intersections et trous, sans créer ni imputer des barres. Funding comporte ses propres timestamps d'événements et ne doit pas être joint comme bougie M5.
- Ne pas confondre mark/index avec un prix spot exécutable et ne pas utiliser le taux de funding réalisé comme signal connu avant son paiement.

Cet audit pilote ne certifie **ni 4 actifs ni 21 mois**, ni les conditions bid-ask/slippage/funding/coût capital. Avant B01/B02, un pipeline immuable multi-source séparé et un protocole économique deux-jambes seront nécessaires.
