# P4 V22 — Protocole gelé de couverture des archives funding/basis

**Gel de conception : 21 octobre 2026, avant tout recensement V22.** L'opérateur a confirmé V21 : 7 tests ciblés réussis, suite complète réussie et ingestion BTCUSDT juin 2025 finalisée avec 8 640 M5 multi-sources et 90 événements de funding. Le succès ne concerne **qu'une crypto et un mois**. Pas de backtest, pas de PnL ni de stratégie B01/B02 à cette étape.

## Univers strict et fontes

- Binance public Data Vision `https://data.binance.vision/data/`, quatre jeux de sources : `spot/monthly/klines`, `futures/um/monthly/klines`, `futures/um/monthly/markPriceKlines`, `futures/um/monthly/fundingRate`. Pour chaque URL réutiliser exactement les templates V19/V21, sans inventer de structure.
- Actifs : **BTCUSDT ETHUSDT SOLUSDT XRPUSDT** et uniquement ces quatre, ni ADA ni autres.
- Mois : **2024-10 à 2026-06 inclus**, 21 mois du développement P4 déjà inspecté. Réserver juillet–septembre 2026 et ultérieur, ne pas le lire.
- Matrice totale préenregistrée : **4 symboles × 21 mois × 4 catégories = 336 archives ZIP**. En mode `sample`, **3 mois** (`2024-10`, `2025-06`, `2026-06`) sur les quatre actifs, soit 48 demandes HEAD. Exécuter le mode `full` uniquement sur commande explicite, après tests et éventuellement sample. Pas de ZIP ni CHECKSUM téléchargés pendant V22.

## Méthodologie

1. Construire un inventaire déterministe local `(symbol, month, kind, url)` sans accès réseau. Refuser les paramètres arbitraires/outputs écrasants.
2. Mode réseau opt-in exclusivement via flag explicite `--probe-public-archives`. Une demande HTTP **HEAD** par ZIP, `timeout<=12s`, `User-Agent` de recherche, 4 requêtes concurrentes maximum; pas de retry non borné, pas de téléchargement, pas de clés API/ordres.
3. Classer la réponse `200` comme `HEAD_OK_UNVERIFIED` uniquement. `404` comme `HEAD_404_NOT_CONFIRMED` (ne pas supposer que le GET échouerait), `403/451/405/autre` et erreurs de réseau `UNVERIFIED`, sans contournement. Les en-têtes `Content-Length` et `Last-Modified` sont de simples métadonnées, pas une preuve d'intégrité.
4. Reporter **tous** les 48/336 couples, y compris les échecs ; éviter qu'un timeout interrompe une couverture partielle sans trace. Rapporter par actif, catégorie et mois, pas seulement un nombre total.
5. Produire un **nouveau JSON et un nouveau CSV hors du dossier P0**, jamais écrasés. Le rapport mentionne explicitement que `HEAD_OK_UNVERIFIED` ne confirme pas le ZIP, son checksum, les bougies, le schéma, la fréquence du funding, les prix spot négociables ou la complétude d'une série.
6. Aucun accès aux rendements ou signaux V14/V17, aucun entraînement ML ou passage d'ordre. V21 sert de preuve de faisabilité du parsing pour BTC 2025-06 ; V22 ne le reprend pas comme preuve des autres mois.

## Décision après recensement

- Si le mode full est entièrement `HEAD_OK_UNVERIFIED`, les 336 **URLs** sont potentiellement accessibles. **Ne jamais dire “336 archives certifiées”** : exiger une phase V23 d'ingestion, de SHA256, de contrôles CSV et de timestamps avec budget réseau/espace disque, reprises et gel des fichiers.
- Si des URL sont absentes ou bloquées, mesurer les trous par type/actif/mois et concevoir l'étude de couverture / biais de survivance avant toute analyse économique. **Ne pas fabriquer ni forward-fill des paiements de funding manquants.**
- La stratégie B01/B02 exige en plus des conventions causales sur funding réalisé versus prévisible, coûts spot/perp des deux jambes, spread/slippage/liquidation et NO_TRADE, préenregistrées avant PnL.

## Résultats V21 servant de point de départ

BTCUSDT juin 2025 : quatre ZIP+CHECKSUM vérifiés, 8 640 M5 spot/perp/mark alignées et contrôlées contre P0, 90 événements funding, ensemble stocké hors P0 dans `p4_v21_btc_2025_06_immutable`. La couverture 84 mois-actifs *n'est pas* démontrée par V21.
