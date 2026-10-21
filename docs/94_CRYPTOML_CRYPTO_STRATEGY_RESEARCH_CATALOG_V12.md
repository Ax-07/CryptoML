# CryptoML — Catalogue de recherche des stratégies crypto, V12

**Créé le 21 octobre 2026, AVANT tout backtest comparatif V12. Ce document est une liste d'hypothèses, PAS un palmarès de stratégies rentables ni une autorisation de trading.** Versions V8 (pullback EMA9/26) négative, V9/V11 (cassure M15/retest M5/H1) non évaluée économiquement. La collecte live V10 reste suspendue.

## Recherche externe, éléments probants et leurs limites

- Liu, Y. & Tsyvinski, A., *Risks and Returns of Cryptocurrency*, Review of Financial Studies 34(6), 2021. Constate du momentum temporel sur crypto ; n'atteste pas les réglages de timeframes ou les rendements de perpétuels avec frais. https://academic.oup.com/rfs/article-abstract/34/6/2689/5912024
- *Intraday return predictability in the cryptocurrency markets: Momentum, reversal, or both*, journal article (2022). Momentum ET renversement intrajournalier conditionnés aux sauts, liquidité et événements ; résultats historiques d'autres marchés/périodes, non transférables tels quels à BTCUSDT perp M5. https://www.sciencedirect.com/science/article/pii/S1062940822000833
- Kitron & Wengrowicz, *Short-horizon mean reversion in cryptocurrency markets* (prépublication, août 2026). Décrit une réversion directionnelle aux 15 minutes sur nombreuses paires ; brut environ 1,3 bp/trade dans la meilleure variante citée, insuffisant face au benchmark 5 bp. https://arxiv.org/abs/2608.21888
- Bui & Nguyen, *Systematic Trend-Following with Adaptive Portfolio Construction* (prépublication, fév. 2026) : étude de trend 6h / gestion de portefeuille, performances déclarées avec hypothèses propres à l'étude ; résultats non considérés comme validés pour CryptoML. https://arxiv.org/abs/2602.11708
- Lesa & Hochreiter, *Cryptocurrency Pair Trading* (2023). Cointegration et réversion de spread ; les coûts et les stops modifient nettement les conclusions. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4433530
- *Pairs Trading Strategies in Cryptocurrency Markets* (2023), 26 cryptos Binance, 79 jours étudiés, différences entre méthodes : étude très dépendante de cette période et du jeu de paires. https://www.mdpi.com/2673-4591/38/1/74
- Gornall et al., *Perpetual Futures and Basis Risk: Evidence from Cryptocurrency* (2025) : structure funding et contrainte de capital ; la valeur relative n'est pas sans risque. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5036933
- *Exploring risk and return profiles of funding rate arbitrage on CEX and DEX* (Blockchain Research and Applications, 2026) : scénarios positifs publiés, mais levier, risque de plateforme, basis et hypothèses limitent toute extrapolation. https://www.sciencedirect.com/science/article/pii/S2096720925000818
- *Centralized–decentralized exchange funding rate arbitrage as a basis trade* (Digital Finance, 2026) : rendements sensibles aux hypothèses, risques de funding, liquidation et intermédiaires. https://link.springer.com/article/10.1007/s42521-026-00213-3
- Brauneis et al., *The crypto world trades at tea time* (2024/2025) : saisonnalités de liquidité, activité et volatilité, pas une preuve de rentabilité d'un simple filtre horaire. https://doi.org/10.1007/s11156-024-01304-1
- Bysik & Ślepaczuk, *Machine Learning-Based Bitcoin Trading Under Transaction Costs* (prépublication, 2026) : signe brut insuffisant pour certaines stratégies à 10bp, sélection tenant compte des coûts parfois meilleure ; mise en garde sur la domination statistique des modèles. https://arxiv.org/abs/2606.00060
- Bailey et al., *The Probability of Backtest Overfitting* (2017) : évaluer beaucoup de variantes sur un même passé augmente le risque de fausses découvertes. https://scholarworks.wmich.edu/math_pubs/42/
- API Binance Futures USD-M : M5 prix/volume/taker_buy, funding, open interest, mark/index, profondeur ; availability et rétention varient par endpoint. Vérifier qualité/provenance avant de joindre. https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data

## Catalogue structuré : 17 familles, aucune présumée gagnante

**Niveau A : faisable sur le P0 vérifié existant (OHLCV M5, volume, trade_count, taker-buy ; recomposition M15/H1/H4/D1).** Niveau A ne signifie ni « edge prouvé » ni premier rang scientifique ; c'est la faisabilité.

| ID | Stratégie et thèse distincte | Déclencheur-type à spécifier avant test | Timeframes candidats | Priorité |
|---|---|---|---|---|
| A01 | **Momentum temporel / suivi de tendance** | retour passé persistant, capter l'extension directionnelle, sans anticiper les ranges | H4 / D1 ; entrée H1 | 1 |
| A02 | **Cassure Donchian / plus-haut plus-bas N périodes** | nouveau haut/bas établi, entrée sur breakout sans attendre retest | H1 / H4 | 1 |
| A03 | **Compression puis expansion de volatilité** | volatilité basse puis breakout d'une borne prédéfinie (squeeze/range) | M15 / H1 | 1 |
| A04 | **Retour à la moyenne après excursion extrême** | éloignement fort d'une référence et signal de stabilisation, stop strict | M15 / H1 | 1 |
| A05 | **V9 : cassure de structure M15 + retest M5 + H1** | reprendre V9 à paramètres déjà gelés, évaluée une seule fois selon V11 | M5 / M15 / H1 | 1, existant |
| A06 | **Momentum avec confirmation de volume / flux taker** | prix directionnel avec accélération volume et taker buy/sell connu à la clôture | M5 / M15 / H1 | 2 |
| A07 | **Réversion en range (Bollinger / canal / VWAP calculable)** | retour vers moyenne lors d'une classe de non-tendance ; éviter transitions | M15 / H1 | 2 |
| A08 | **Momentum relatif inter-cryptos** | classer performances passées BTC/ETH/SOL/XRP et suivre fortes/faibles valeurs | H4 / D1 | 2 (univers petit) |
| A09 | **Réversion après excès impulsif** | grande bougie + pic de volume, puis retour opposé, distinct du simple « contre bougie » | M15 / H1 | 2 |
| A10 | **Pullback en forte tendance H4/H1 sans EMA9 M5** | continuation après correction de structure, en gardant un signal distinct de V8 | H1 / H4 | 3 |
| A11 | **Filtre d'horaire / liquidité** | modulation préfixée de A01/A02 ; pas une « stratégie gagnante » autonome | UTC M15/H1 | auxiliaire |

**Niveau B : nouvelles données synchronisées nécessaires ; construire une infrastructure de provenance indépendante AVANT de tester.**

| ID | Stratégie et moteur économique | Données en plus | Priorité |
|---|---|---|---|
| B01 | **Carry funding delta-neutre spot long / perpetual short** | spot + perp, funding réel, frais des deux jambes, basis, marges | 1B |
| B02 | **Réversion du basis spot/perp** | synchronisation spot/mark/index/perp, funding et exécution bilatérale | 1B |
| B03 | **Pairs cointegrés / spread stat-arb** | séries simultanées des deux actifs, hedge ratio entraîné causalement, coûts deux jambes | 2B |
| B04 | **Extrêmes de funding + positionnement contrarian** | historique funding pré-connu et OI/ratios valides, pas seulement retour de prix | 2B |
| B05 | **Open interest et liquidations (« squeeze ») comme contexte** | OI historique fiable, prix/mark, éventuels flux liquidation horodatés | 3B |

**Niveau C : reporté en raison des coûts et données microstructure ; pas dans première vague.**

| ID | Stratégie | Raison du report |
|---|---|---|
| C01 | **Order-book imbalance / market making / microstructure L2** | nécessite snapshots + mise à jour carnet, latence, adverse selection, frais maker/taker, files d'attente ; M5 incapable de prouver les fills |

### Deux distinctions importantes

1. La **classe RANGE** V6 est une classe *résiduelle* qui inclut transition/désaccord et ne suffit pas à prouver qu'une réversion A07 fonctionne. Il faudra figer une définition testable différente et enregistrer ses hypothèses.
2. A01, A02, A03, A04 sont de véritables mécanismes différents. Des variantes EMA/RSI/MACD/Bollinger qui testent toutes la même logique ne sont **pas** 4 découvertes indépendantes.

## Ordre de travail recommandé

**Lot I** : A01 momentum H4/D1, A02 Donchian H1/H4, A03 compression-breakout, A04 réversion sur choc M15/H1, et A05 V9 déjà codé. Objectif premier : estimer **brut/trade**, turnover, potentiel de mouvement et distance aux frais. Si brut insuffisant, arrêter la famille sans lui appliquer un filtre IA.

**Lot II** : A06 volume/taker, A07 range mean reversion, A08 cross-sectional, A09 excès, A10 continuation structurelle. Ne lancer que si des observations de Lot I indiquent pourquoi les améliorations prévues seraient économiquement utiles ; ils restent des tests nouveaux, pas des optimisations opportunistes.

**Lot III**, indépendant : B01/B02/B03, si des données funding/spot historiques fiables peuvent être acquises. B04/B05 demandent davantage de provenance et peuvent être indisponibles sur 21 mois via endpoints (certaines stats Binance ne conservent que 30 jours). C01 hors périmètre actuel.

## Règles de recherche non négociables

1. **Aucune implémentation/backtest du catalogue maintenant** : d'abord fixer un protocole unitaire propre à chaque famille, une spécification de signal et un coût de transaction réaliste. Ce document ne gèle PAS les hyperparamètres concrets ; il gèle la liste des familles et les statuts de faisabilité.
2. Tests histo sur P0 approuvé (octobre 2024–juin 2026) purement **exploratoires** car le passé a été inspecté ; juillet–septembre 2026 également déjà exploré dans CryptoML, **pas un holdout nouveau**.
3. Fixer la période de calibration ET la période d'évaluation de façon causale *dans l'historique*, puis étiqueter le score `HISTORICAL_EXPLORED_NOT_INDEPENDENT`. Ne jamais appeler cela OOS indépendante. Une évaluation en temps réel/papier n'est PAS requise pour le simple diagnostic initial, mais une validation prospective indépendante serait requise avant revendication de rentabilité.
4. Aucune grille massive : une version de référence prédéfinie par famille ; éventuelle sensibilité prédéclarée avant issue ; comptabiliser **tous** les essais et variantes dans un registre de recherche. Une des 17 familles finira peut-être gagnante par hasard ; corriger cette sélection multiple.
5. Simulateur commun : LONG/SHORT symétrique, horodatages au close, exécution retardée, stop prioritaire OHLC ambigu, gaps réalistes, coûts explicites 2/5/10 bp comme sensibilité **non garantie**, funding et frais de financement si pertinents, pas d'ordre en réel.
6. Rapporter coverage, espérance brute et nette par trade, somme notionnelle et performance portefeuille uniquement si un portefeuille réel simulé avec conflits/expositions/capital/funding, distribution, incertitude et stabilité temporelle. Une moyenne positive n'est pas une preuve statistique de robustesse.
7. Comparateurs ZERO_TRADE, buy/hold ou short-and-hold selon stratégie, direction, stratégie triviale coût comparable, calendrier/durée identiques.
8. Ne pas rebaptiser résultats négatifs. Ne pas augmenter le nombre de filtres jusqu'à « trouver » une performance. Documenter les échecs comme résultats de recherche.
9. Le modèle d'IA arrive **après** diagnostic de mécanisme et seulement si une justification hors cible suggère qu'il peut sélectionner ou dimensionner des signaux au-delà des coûts.
10. **La V11 préexistante ne doit pas être lancée avant qu'une règle d'évaluation commune au Lot I soit arrêtée** : dans ce catalogue V9 sera évaluée avec les autres et documentée comme l'hypothèse suggérée précédemment.

## Prochaine décision

Concevoir et figer un **protocole comparatif Lot I (A01–A05)** avec horizons suffisamment longs pour diagnostiquer le brut, politique commune de frais/funding et règles anti-fuite. **Aucun paramètre n'est encore choisi en fonction de performances V12.** Le but est de chercher un avantage, pas de valider une stratégie prédéterminée.
