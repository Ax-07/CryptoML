# P4 V18 — Clôture V17 A06 et faisabilité funding / basis, recherche sans trading

Date 2026-10-21. Résultats recopiés de la console de l'opérateur après réussite de 11 tests V16/V17 et de la suite complète. V16 a réconcilié 8 833 événements, V17 a affiché `V16 FINGERPRINT + ALL EVENTS MATCH`. Les fichiers locaux JSON et CSV ne sont pas présents sur GitHub : pas de recalcul indépendant revendiqué.

## Verdict A06 pré-enregistré

| Sens | Signaux maturés | Brut moyen (bp) | Net après 2 bp | Net après 5 bp | Net après 10 bp |
|---|---:|---:|---:|---:|---:|
| LONG | 3827 | +0.916207 | −1.083793 | −4.083793 | −9.083793 |
| SHORT | 5006 | +1.526690 | −0.473310 | −3.473310 | −8.473310 |
| Total | 8833 | +1.262191 | −0.737809 | −3.737809 | −8.737809 |

**A06 configuration fixée échoue même au plancher supposé de 2 bp : STOP**, sans optimiser les seuils 0,20 / 1,5 / 0,35 ou l'horizon de 12 M5 sur ces 21 mois. Aucun rendement économique indépendant démontré sur les six stratégies étudiées. Les moyennes restent des markouts individuels sans funding, slippage ou portefeuille.

## Hypothèse B01 / B02 — Économie delta-neutre, données distinctes

**B01** long spot/short perp sur funding positif (ou construction inverse lorsque le financement négatif et la disponibilité d'emprunt le permettent), avec couverture de direction réelle imparfaite. Le funding historique *réalisé* ne peut être utilisé comme information disponible AVANT la décision, sauf données qui prouvent son annonce antérieure ; ce sont les paiements observés après coup. B01 suppose deux jambes, risques de basis, funding variable, frais deux marchés, slippage, liquidité, marge, liquidations, risque de plateforme et coût/risque d'emprunt selon direction.

**B02** basis spot/perp : un prix mark perp n'est ni un cours spot négociable ni une garantie de fill. Synchroniser prix spot négociables et perpétuels, quotes bid/ask si disponibles, horodatages UTC, funding daté, coût des deux jambes et clôture cohérente. Aucun arbitrage « sans risque » revendiqué.

L'API REST de taux réalisés Binance Futures USD-M `GET /fapi/v1/fundingRate` admet `symbol`, `startTime`, `endTime`, `limit` (jusqu'à 1000); son existence ne garantit pas la disponibilité ni l'exhaustivité de 2024-10 à 2026-06. Certains endpoints de statistiques de positionnement/basis n'ont que 30 jours de rétention, donc ne peuvent pas remplacer l'historique brut. **Les archives Binance Data Vision peuvent offrir d'autres couvertures, à vérifier et contrôler SHA-256 plutôt que présumer.**

Références externes :
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Basis
- https://github.com/binance/binance-public-data
- https://data.binance.vision/

## V18 : audit des sources uniquement — aucun calcul de rendement

Un contrôleur **opt-in** réalisera une sonde réseau read-only bornée pour :
- archives publiques mensuelles `spot/monthly/klines/<SYM>/5m`, `futures/um/monthly/klines/<SYM>/5m`, `futures/um/monthly/markPriceKlines/<SYM>/5m`, `futures/um/monthly/fundingRate/<SYM>` si la structure annoncée correspond ;
- mois échantillons 2024-10 / 2025-06 / 2026-06 ; **la simple présence d'un échantillon n'atteste jamais 21 mois ni leur intégrité** ;
- taux de funding sur une fenêtre ancienne via API GET public, sans clé, sans proxy de contournement ; erreurs HTTP/réseau documentées comme indisponibilités/non vérifications ;
- les réponses d'archive ne sont pas téléchargées ni ingérées dans P0 dans cette étape. Pour production future : ZIP, .CHECKSUM, manifeste, validation des schémas de temps et fiscalité exacte.

Les scores V14/V17 ne sont **pas** comparables à la rentabilité nette d'un carry spot-perp. La politique d'évaluation des deux jambes devra être préenregistrée après preuve de couverture ; absence de portefeuille réel/portefeuille live avant validation prospective indépendante.
