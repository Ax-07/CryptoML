# P4 V13 — Lot I multi-stratégies : protocole de recensement SANS performances

**Gelé le 21 octobre 2026 avant tout recensement V13 et avant toute évaluation économique comparative.** V8 ne montre aucun edge net pour le pullback EMA9/EMA26. V9/V11 n'a pas encore subi le backtest économique réel. V10 et son observation en direct restent suspendus. Le but à terme est un *méta-modèle* qui sélectionne la stratégie offrant le meilleur rendement ajusté du risque après frais ou **NO_TRADE**, non une stratégie unique.

## Cadre commun, causal, sans optimisation

Univers exclusif : BTCUSDT / ETHUSDT / SOLUSDT / XRPUSDT, 84 Parquets P0 contrôlés SHA-256 et qualité, octobre 2024 à juin 2026. Juillet–septembre 2026 et ADA exclus. Les données historiques sont **déjà explorées**, donc aucune validation indépendante sur cette période. M5 Binance Futures clôturées, H1 = 12 M5, H4 = 48 M5 ; agrégation complète alignée UTC, interdiction de remplacer les trous par des bougies artificielles, réinitialisation des indicateurs dès rupture de continuité. Tous les signaux sont calculés **à la clôture** d'une bougie complète, et leur prix de référence est un prix connu au signal.

**V13 ne crée aucun résultat économique** (pas de PnL, retours futurs, frais, stops, labels ni optimisation ML). Il compte les opportunités et produit un registre de signaux horodatés reproductible. Fixer l'ordre de traitement des stratégies et les règles de sélection du futur méta-modèle dans une phase distincte avant tout entraînement.

## Les cinq familles et leurs règles spécifiques gelées

### A01 — Momentum H4, contrôle directionnel quotidien

Au close H4 exactement à **00:00 UTC**, calculer le log-return cumulatif des **24 H4** clôturées et la volatilité réalisée (écart-type, ddof=0) des **24 log-returns H4**, multipliée par √24. LONG si score return/volatilité >= +1,00 ET close H4 > moyenne mobile simple 24 H4 ; SHORT si score <= −1,00 ET close H4 < SMA24. Pas de signal autrement. Une observation journalière maximum par actif. Ce test porte un momentum de plusieurs jours, non le pullback EMA9/EMA26 M5.

### A02 — Donchian H1

Plus haut des HIGH des **20 H1 précédentes**, plus bas des LOW des **20 H1 précédentes**, exclure impérativement H1 du signal. LONG si close H1 > haut + **0,10 × ATR14 H1**, SHORT si close H1 < bas − 0,10 × ATR14 H1. ATR14 calculé uniquement sur H1 clôturées et resets aux trous. **Cooldown 12 H1** par stratégie sur un même actif, peu importe le sens (pas d'exécution réelle).

### A03 — Compression puis breakout H1

Plage des **12 H1 précédentes** (= high max − low min) divisée par ATR14 connu à la **précédente H1**. Cette largeur doit être <= **0,70 × la médiane des 48 valeurs historiques antérieures de la même largeur**, toutes calculées sans la H1 du signal. Confirmation : au close H1 du signal LONG au-dessus du haut des 12 H1 précédentes +0,10 ATR14(H1,t), SHORT au-dessous du bas −0,10 ATR14(H1,t). **Cooldown 12 H1**, tous sens confondus. Pas de paramètre modifié sur la base du nombre de signaux.

### A04 — Réversion H1 après excursion

Bande Bollinger simple sur les **20 H1 clôturées**, moyenne±**2 écarts-types** (ddof=0). Mesurer les bandes à **t−1** : LONG si close(t−1) < bande basse(t−1), puis close(t) > bande basse(t−1), close(t) < SMA20(t−1) et close(t) > open(t) ; SHORT symétrique, passage sous la bande haute de t−1 et bougie baissière. Condition de faible directionnalité : **ER12 de t−1 <=0,30** (|close(t−1)−close(t−13)| / somme absolue des 12 mouvements). **Cooldown 12 H1**, tous sens confondus. La classe RANGE V6 n'est pas interprétée comme vrai range.

### A05 — Cassure M15 et retest M5/H1 V9

Mécanique V9 existante, **sans nouveau seuil**. La copie V11 `structure_breakout_retest_historical_v11.detect` peut être appelée uniquement en recherche rétrospective, avec un start de recherche en octobre 2024 ; la détection `structure_breakout_retest_v9.detect` prospective reste inchangée. Cassure 0,1 ATR, plage des 12 M15 préalablement closes, retest dans 6 M5, invalidation 0,75 ATR, classe H1 UP/DOWN, cooldown 8 M5. Le label `OFFLINE_REPLAY_NOT_LIVE_ATTESTED` est obligatoire.

## Contrôles obligatoires et sortie

1. Test de symétrie LONG/SHORT, indisponibilité des bougies incomplètes, trous et reset, références temporelles strictement antérieures, unicité du signal par stratégie×actif×close.
2. Test de non-fuite : les signaux jusqu'à `t` doivent rester strictement identiques lorsque seules les bougies **après t** changent.
3. Rapport JSON : protocol SHA version, empreinte des manifests, 4 symboles, bornes, nombre M5 vérifiées, nombres par stratégie/actif/sens/mois, comparateur NO_TRADE conceptuel.
4. CSV registre par signal : stratégie, actif, side, timeframe, horodatage de décision, prix de référence connu, ATR connu ; sans outcome. CSV comptes mensuels ; **trois nouveaux fichiers**, hors P0.
5. Interdiction de joindre des retours ou futures OHLC, des labels V8, les positions futures ou les données déjà réservées 2026-07/09.
6. Aucune suppression d'une stratégie si les signaux sont peu nombreux : publier zéro si applicable. Si les tests révèlent une anomalie, corriger **avant de révéler des performances**, en documentant la correction.

## Plan après census

V14 devra d'abord figer une **politique économique commune** appropriée aux cadences M5/H1/H4 : bar d'entrée réelle post-décision, sorties, horizon indépendant des résultats, funding, coûts, concurrence des signaux, benchmark par temps en position et NO_TRADE. Il est **interdit** de choisir l'horizon ou les coûts par stratégie d'après leurs backtests. Développer ensuite un même moteur de recherche et un registre exhaustif d'expériences. Une performance positive sur le passé inspecté reste exploratoire et ne valide pas le futur méta-modèle.

Statut initial : `MULTISTRATEGY_CANDIDATE_CENSUS_ONLY`, non trading, non IA entraînée.
