# P4 V20 — Vérification ciblée du schéma de funding, **sans PnL**

**Préenregistrement : 21 octobre 2026.** Pas de trading, ni backtest, ni modèle ML, ni nouvelle partition P0. V19 a validé les ZIP / SHA256 et les bougies M5 de BTCUSDT juin 2025 : spot, USD-M perp et mark **8 640 bougies chacun** ; le schéma funding restait `SCHEMA_REVIEW_REQUIRED`. L'alignement exact entre les trois séries doit être lu et vérifié dans le JSON V19 : les trois nombres de lignes seuls ne le prouvent pas.

## Objectif et périmètre

Valider les champs et les horodatages des paiements de funding BTCUSDT juin 2025 dans **l'archive officielle `futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2025-06.zip`**. L'archive `.CHECKSUM` SHA-256 est une condition impérative. Deux requêtes publiques explicites (ZIP et CHECKSUM), aucun ordre, aucune clé API. Le programme doit lire le rapport V19 local comme référence et **refuser** toute erreur / source insuffisamment validée. Aucun prétendu historique complet 4 actifs × 21 mois.

## Parsing fail-closed, schémas explicites

Les travaux externes citent des champs archives `calc_time`, `funding_interval_hours`, `last_funding_rate` (structure **hypothétique** à vérifier par les vrais octets). Accepter cet ensemble de colonnes seulement s'il apparaît exactement ; une variante `fundingTime`, `fundingRate` sera acceptée si le vrai CSV la porte. Pour tout autre schéma : `SCHEMA_REVIEW_REQUIRED` et liste de colonnes observées sans extrapolation. Références de recherche uniquement : https://github.com/binance/binance-public-data et https://github.com/nhocconan/AutoResearch-based-Trading-Strategy-Generation-and-Testing/blob/master/docs/data-pipeline.md

- CSV unique ZIP, nom non dangereux, limites taille zip/décompressé/ratio, checksum exact ;
- timestamps d'**événement de funding** numérique en secondes, millisecondes ou microsecondes, **sans mélange**, convertis UTC, strictement croissants et uniques, de 2025-06-01 inclus à 2025-07-01 exclu ;
- taux de funding numériques et finis, unités **fraction décimale** telles que fournies : une valeur `0.0001` signifie 1 bp de taux, et non 0.0001 bp. Rejeter les formats en pourcentage, les valeurs non plausibles et les NaN au lieu d'estimer un taux ;
- vérifier `funding_interval_hours` si présent (entiers positifs bornés). Les intervalles de financement peuvent changer ; ne pas forcer un paiement toutes les huit heures et ne pas utiliser un taux payé au temps `t` comme information antérieure à t ;
- statut `FUNDING_SCHEMA_AND_CHECKSUM_OK` seulement si **tous** les contrôles réussissent ; toute structure inconnue reste bloquante.

## Liens et limites

Le JSON final doit donner : SHA du rapport V19, SHA du ZIP de funding, colonnes observées, unité des timestamps, nombre d'événements, premier/dernier événement, volumes / nombres de lignes et chevauchement M5 issus du V19, sans exposer les taux individuels ni leurs retours. L'intégrité des fichiers ne démontre ni **la connaissance causale d'un funding futur**, ni la négociabilité du basis ; aucune rentabilité de B01/B02 n'est calculée.

Une étude de 4 actifs × 21 mois exige ensuite de vérifier la couverture complète et l'identité des sources spot/perp/mark/funding, d'ingérer les données dans un espace distinct protégé et de **préenregistrer** les deux jambes spot/perp, prix exécutables, spreads/slippage, financement effectif, collatéral, liquidation, borrow/rebalancing et risques de plateforme **avant** toute lecture de rendements.
