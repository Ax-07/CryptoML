# P4 V3 — Clôture des Ridge 18 mois (20 octobre 2026)

**Décision : `NO_ELIGIBLE_TRADING_STRATEGY`. Aucun réajustement autorisé sur les diagnostics de mai–juin 2026.**

Source : sortie PowerShell communiquée par l'opérateur ; les fichiers locaux JSON et CSV complets n'ont pas été audités ici. Les données historiques de mai–juin ont déjà servi à des évaluations exploratoires et ne sont pas un OOS indépendant.

## Contrôles déclarés

- 7 tests synthétiques ciblés passent et la suite `pytest -q` ne présente aucun échec ; avertissements Starlette/httpx et LightGBM uniquement.
- Chargement P0 SHA-256 depuis 2024-10 à 2026-06, uniquement BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT ; ADA et juillet–septembre 2026 exclus.
- Signaux détecteur V1 : BTC 5480, ETH 5432, SOL 5368, XRP 5303, soit **21 583**.
- Trades A mûrs : BTC 5480, ETH 5432, SOL 5367, XRP 5303, soit **21 582**. Un signal SOL n'avait pas d'issue mûre ; la cause précise n'est pas déterminée par le log.
- Confirmation B, sur 21 mois : BTC 2330, ETH 2374, SOL 2304, XRP 2230 ; **9 238** au total.
- Seuils de décision des trois Ridge : `0.0` (règle stricte de score prédit net 5bp > seuil).

## Diagnostics mai–juin 2026 — rendement théorique après 5 bp aller-retour

| Politique | B train | Trades | bp/trade | bp/opportunité | Écart bp/trade vs random B à même effectif actif×mois |
| --- | ---: | ---: | ---: | ---: | ---: |
| A all | — | 1887 | −5.313285162932595 | −5.313285162932595 | — |
| B confirmed | — | 825 | −5.681926572457758 | −2.4841491374020404 | — |
| C0 Ridge9 train 4 mois | 1649 | 84 | −7.927495340096827 | −0.3528932742809398 | −2.2614693269124597 |
| C1 Ridge9 train 18 mois | 8002 | **0** | sans objet | **0.0** | sans objet |
| C2 Ridge19 train 18 mois | 8002 | **0** | sans objet | **0.0** | sans objet |
| D never trade | — | 0 | sans objet | 0.0 | sans objet |

L'abstention C1/C2 n'établit **aucun avantage économique**, puisque aucun trade n'a été observé. C0 est moins favorable que l'espérance *analytique* de tirage aléatoire avec les mêmes nombres de trades par actif×mois ; ce n'est ni une permutation effectuée ni une p-value. Le net B moyen à 5 bp implique un brut théorique d'environ **−0,6819 bp/trade**, ce qui suggère d'examiner l'économie de la stratégie mécanique plutôt que de retuner le modèle.

## Gel et prochaine recherche

- Ne pas modifier les paramètres V3 ou sélectionner les meilleurs mois, symboles, classes ATR, stops ou seuils à partir de ces résultats.
- Audit descriptif **préspécifié**, sur une branche séparée, des labels A/B de développement (distribution du gross, catégories d'exit, impact arithmétique de coûts 2/5/10bp et concentration par mois/symbole). Ne pas utiliser cet audit pour lancer des optimisations a posteriori.
- Toutes les sorties originales restent immuables. Pour une réelle confirmation, politiques verrouillées et **prédictions futures** horodatées avant observation, avec évaluations de coûts/fills réels.

**Statut : `RESEARCH_CLOSED_NO_ELIGIBLE_STRATEGY`.**
