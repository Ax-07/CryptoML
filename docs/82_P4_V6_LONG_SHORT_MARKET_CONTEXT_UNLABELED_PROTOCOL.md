# P4 V6 — Reconnaissance causale des marchés et candidats LONG / SHORT

**Statut : protocole descriptif fixé avant tout résultat V6. Aucun entraînement, aucun label, aucun backtest, aucun ordre.** Branche indépendante `research/market-structure-long-short-v6`; le pullback exclusivement LONG V1–V5, ses résultats et sa clôture demeurent inchangés.

## Pourquoi cette hypothèse ?

L'examen visuel du trade ETHUSDT du 6 juin 2026 a suggéré qu'un croisement ou des pentes d'EMA9/EMA26 M5 peuvent survenir à l'intérieur d'un **range**. Ce diagnostic est une hypothèse issue d'un événement déjà inspecté, **non une preuve** ; ne pas utiliser ce trade comme holdout pour la suite.

Le marché source est **Binance USD-M perpetual**, mais l'ancienne stratégie n'émettait que des LONG par conception. Cette V6 recense explicitement les deux directions sans jamais placer d'ordre.

## Définition gelée du contexte, à la clôture des bougies seulement

À partir de M5 finalisées SHA-256 :
1. Construire M15 (exactement trois M5 successives) et H1 (douze M5 successives), alignées UTC. Exclure toute bougie supérieure incomplète ; ne jamais interpoler ni inventer un close H1 avant qu'il soit connu. Toute discontinuité reinitialise les indicateurs.
2. Pour **chaque** timeframe, calculer EMA9, EMA26, ATR14, pente EMA9/EMA26 sur trois bougies complètes et ratio d'efficacité directionnelle ER12 = |close(t)−close(t−12)| / somme des 12 variations absolues. Les seuils sont des hypothèses de recensement, pas des paramètres optimisés.
3. Contextes `UP` : écart EMA9−EMA26 ≥ +0,35 ATR, deux EMA en hausse, close > EMA26, ER12 ≥ 0,35. `DOWN` : strict symétrique (écart ≤ −0,35 ATR, pentes négatives, close < EMA26). `RANGE` : historique suffisant mais aucune des deux tendances ; étiquette *résiduelle*, qui inclut certaines transitions. `UNKNOWN` : historique incomplet ou invalidé.
4. À chaque clôture M5, joindre **uniquement** le dernier M15 et H1 totalement clôturé, et aucun état dont la clôture est âgée d'un timeframe entier. `UP` global si M15 et H1 sont UP ; `DOWN` si les deux DOWN ; `UNKNOWN` si l'un est indisponible ; sinon `RANGE` (incluant désaccord/transition). Ce sont des classes de recherche, **pas une reconnaissance garantie du vrai régime de marché**.

## Signaux non labellisés, logique symétrique

- `LONG` : mêmes règles historiques de premier retour sur EMA9 après hausse EMA9 > EMA26 et pentes positives, clôture préservant la structure / ATR. Cooldown huit bougies par direction.
- `SHORT` : miroir exact (EMA9 < EMA26, pentes négatives, `high(t)` touche EMA9 depuis dessous, close ≤ EMA26, dépassement EMA26 limité à 0,25 ATR). Cooldown huit bougies par direction.
- `long_with_trend` = signal LONG et régime global UP ; `short_with_trend` = signal SHORT et régime global DOWN. Ce **ne sont pas des ordres de trading ou filtres validés économiquement**.
- Tous les candidats de chaque sens et leur répartition UP/DOWN/RANGE/UNKNOWN sont publiés *avant d'ouvrir les outcomes*. Aucun seuil, actif, mois, timeframe ou signal ne sera sélectionné sur la base d'une performance découverte a posteriori.

## Périmètre immuable

- BTCUSDT / ETHUSDT / SOLUSDT / XRPUSDT, octobre 2024 → juin 2026 uniquement : 84 Parquets P0, SHA-256 et qualité vérifiés par le chargeur V3 existant.
- Aucun Parquet ADA, juillet–septembre 2026 ou résultat P3. Périodes déjà explorées donc descriptives, **non OOS indépendant**.
- Sortie fixe JSON + CSV nouveaux, comptes par mois et crypto, états et directions. Pas de label, trade, frais, profit, backtest ou réentraînement dans la V6.

## Windows

```powershell
cd E:\cryptoML-pullback-trade-charts-v5
git fetch origin
git worktree add E:\cryptoML-market-structure-v6 origin/research/market-structure-long-short-v6
cd E:\cryptoML-market-structure-v6\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_market_structure_v6.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

# UNIQUEMENT si les deux tests sont verts
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.market_structure_census_v6 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v6_market_structure_census_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v6_market_structure_census_v1.csv"
```

Ces deux fichiers doivent être **nouveaux** et extérieurs à P0. L'analyse H1+M15 de 21 mois M5 peut être plus coûteuse que les recensements V2/V3, car chaque timeframe est reconstituée et testée pour sa continuité.

## Interprétation et phase suivante

Première phase : uniquement compter la disponibilité des quatre contextes et des candidats dans chaque sens. Si `UNKNOWN` est élevé, **auditer les trous de données et l'initialisation**, ne pas abaisser des critères après l'avoir vu. Si `RANGE` domine, ne pas déclarer l'algorithme « mauvais » : la définition est une hypothèse stricte de tendance, pas une vérité empirique.

Une phase d'issues LONG/SHORT et d'apprentissage nécessitera **un autre document gelé, un autre module et des tests spécifiques** (horloges t/t+1/t+2, stops, frais, funding, markouts, concurrence entre LONG et SHORT) avant d'examiner ses rendements. La comparaison économique sur historique déjà inspecté restera exploratoire ; validation éventuelle sur **prédictions réellement futures** verrouillées en avance.
