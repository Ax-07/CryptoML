# P4 V16 — Protocole gelé : données taker/volume et faisabilité funding/basis

**Date de préenregistrement : 21 octobre 2026, AVANT toute analyse de performances d'A06.** Ce document n'est pas un backtest, ni une affirmation de rentabilité. Les cinq signaux V13/V14 ne sont pas modifiés ; V15 a montré A01 Momentum H4 net moyen +0,0053 bp à 5 bp (économiquement nul), très instable selon actifs et mois, et quatre autres configurations négatives avant frais.

## Sources et données disponibles

- Les 84 Parquets P0 Binance USDⓈ-M perpetual M5 immuables et contrôlés : BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT, 2024-10 à 2026-06 seulement. `ingestion.schema.COLUMNS` inclut **volume_base, volume_quote, taker_buy_volume_base, taker_buy_volume_quote, trade_count** ; les expériences V13 n'en ont pas exploité le flux agressif.
- Le chargeur `load_long_development_history` vérifie SHA-256, qualité, finalité, UTC et géométrie; il exclut ADA et juillet–septembre 2026. Il reste **historique déjà étudié, pas OOS indépendant**.
- API officielle Binance Futures USDⓈ-M : `GET /fapi/v1/klines` contient les volumes taker par bougie; `GET /fapi/v1/fundingRate` retourne des **taux réalisés** avec `fundingTime` et `markPrice` ; `GET /fapi/v1/markPriceKlines` fournit un *mark*, **non un prix spot exécutable**. Les endpoints statistiques `/futures/data/openInterestHist`, `/futures/data/takerlongshortRatio` ne donnent que 30 jours de disponibilité ; ne pas les substituer aux champs des bougies P0 ni prétendre reconstruire 21 mois d'OI.
  Références : https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history ; https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#kline-candlestick-data ; https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#open-interest-statistics

## Hypothèse A06 — impulsion confirmée par volume et takers

À la clôture de **chaque M15 constituée exactement de 3 M5 closes et consécutives** (aucune barre synthétique ni période courante incomplète), calculer :
1. volume agressif fraction `taker_buy_volume_base / volume_base` et déséquilibre `I = 2 × fraction − 1`, pour `volume_base > 0`.
2. ratio volume `volume_base(M15,t) / médiane(volume_base(M15,t−32…t−1))`, toutes ces M15 consécutives au sein du segment, médiane strictement positive. Aucun volume ultérieur utilisé.
3. ATR14 M15 causal issu de `_segment_features` réinitialisée aux trous, strictement positif.

**LONG** au close M15 si `I >= +0,20`, `volume_ratio >= 1,50`, `close > open`, `close-open >= 0,35×ATR14`. **SHORT** symétrique : `I <= −0,20`, `volume_ratio >= 1,50`, `close < open`, `open-close >= 0,35×ATR14`. **Cooldown 6 M15** sur un même actif, tous sens, réinitialisé à chaque gap. Aucun changement de seuil selon les résultats, aucun choix post hoc d'actifs/directions.

Ce signal ne garantit pas que l'achat agressif crée un edge : le flux taker et le mouvement de prix du M15 peuvent être fortement corrélés. V16 est uniquement **l'audit de qualité et le recensement SANS LABELS, sans rendements, sans PnL**.

## Contrôles du P0

Exiger données OHLC finalisées et vérifiées. Volumes et trade_count numériques, finis, positifs ou nuls ; `taker_buy_volume_base <= volume_base + tolérance de flottant` et idem quote ; volume taker non négatif. Compter les bougies volume nul, les fenêtres M15 exclues et les signaux par symbole/sens/mois. Si anomalie de qualité, **arrêter l'audit, ne pas corriger P0 à la volée**. Ne pas déduire des rachats BTC réels, du funding ou de l'OI depuis taker buy volume.

## Funding/basis : première sonde autorisée mais NON implicite

Le programme est **hors réseau par défaut**. L'option `--probe-funding-history` réalise, uniquement avec GET public non authentifié, une requête bornée de taux **déjà réalisés** pour la fenêtre UTC `2025-06-01 00:00`–`2025-06-03 00:00` sur chacun des quatre symboles (4 requêtes maximum). Une réponse non vide démontre uniquement la disponibilité *d'un échantillon* et **pas** la totalité des 21 mois. La sonde ne sauvegarde aucun historique financé comme signal et ne calcule aucun rendement ; un HTTP 403/451, timeout, tableau vide ou schéma inattendu doit être reporté comme non vérifié/indisponible, sans contournement réseau.

**Carry B01 LONG spot / SHORT perpetual** : il faut deux jambes réellement négociables, leurs prix et exécutions historiques synchronisés, funding daté, frais/rebalancements, basis et marge; funding positif peut payer les short perp mais le taux futur est inconnu à t. Un taux réalisé n'est JAMAIS une feature prédictive disponible avant son timestamp. Funding négatif peut inverser le cashflow. **B02 basis mean reversion** exige également des prix spot négociables synchronisés ; le mark/index Binance n'est pas le spot exécutable. L'étude B01/B02 reste BLOQUÉE jusqu'à preuve d'existence, qualité et couverture historique de ces séries et d'une politique de coûts/calendrier.

## Outputs et décision

Sorties nouvelles JSON + CSV de signaux A06 **hors P0**, SHA fingerprint 84 partitions ; aucune écriture dans les anciens Parquets ou les résultats V13–V15. Signaux triés avec horodatage de décision, référence close connue, ATR, ratio volume, déséquilibre. Sommaires par symbole/mois/sens, compte de M15 complètes, volume nul, taux de couverture. Tests synthétiques : géométrie, contrôle volume taker, reset trous M5/M15, invariance aux bougies futures, symétrie LONG/SHORT, non-doublons, refus d'outputs déjà présents.

Après V16, **avant tout backtest A06** : définir et geler horizons, barrières, coûts réels/perp funding et comparaisons ; accepter résultat négatif sans réoptimiser 0,20/1,50/0,35. V16 n'autorise aucun ordre réel, aucune tâche live, aucun entraînement du méta-modèle.
