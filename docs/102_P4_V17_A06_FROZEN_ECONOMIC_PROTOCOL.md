# CryptoML P4 V17 — Audit économique **pré-enregistré** de l'impulsion taker/volume A06

**Gel du protocole : 21 octobre 2026.** Aucun retour, résultat de backtest ni fichier d'outcomes A06 n'a été consulté pour fixer les paramètres ci-dessous. Il s'agit d'un audit historique **exploratoire sur une période déjà observée**, et non d'un nouveau test hors échantillon indépendant. Aucun ordre réel, paper live, entraînement de méta-sélecteur ou optimisation des paramètres.

## Données validées et signaux gelés

Le V16 a passé ses 5 tests spécifiques, la suite complète, les contrôles de champs volume/taker des 84 partitions P0 immuables, et a recensé 8 833 signaux : BTC 1327 LONG/1386 SHORT, ETH 1057/1225, SOL 856/1274, XRP 587/1121. **Aucun résultat économique observé.**

Le détecteur est **strictement** `mlpipeline.volume_funding_readiness_v16.census_symbol`, non modifié :
- M15 constituées de trois M5 closes consécutives ;
- impulsion directionnelle >=0,35 ATR14 M15 ; pression `2 * taker_buy_volume_base / volume_base - 1` >=+0,20 pour LONG ou <=−0,20 pour SHORT ;
- volume M15 >=1,5 fois la médiane des 32 M15 précédentes ;
- cooldown 6 M15, segmentation aux trous. Le signal n'est connu qu'à la clôture M15 ; aucune donnée du futur.

Le module V17 devra charger à nouveau les 84 Parquets P0 **vérifiés SHA-256**, et comparer le `fingerprint_sha256`, `protocol`, `symbols`, `source_partitions`, l'inventaire des signaux par actif et les **8 833 événements horodatés et leurs champs numériques** aux deux fichiers V16 d'origine, avant toute mesure économique. Si anomalie de référence ou de cohérence : ARRÊT, pas de résultats.

## Simulation prédéfinie, non calibrée sur les gains

- **Direction :** symétrique LONG et SHORT, sur les quatre actifs, sans exclusion post-hoc.
- **Entrée hypothétique :** `OPEN` de la première M5 contiguë **après** la clôture M15 signal. Aucune entrée rétroactive à l'intérieur du M15 de signal.
- **ATR :** `atr14_m15` gelé au signal, strictement positif/finie.
- **Stop/target :** distance stop `1×ATR14(M15)`, target `1,5×ATR14(M15)`, symétriques en LONG/SHORT. Utiliser strictement la mécanique `_barrier` existante V8 (gap défavorable à l'open, gap favorable target plafonné, stop avant target si les deux sur une même M5).
- **Horizon de sortie primaire UNIQUE : 12 bougies M5 = 60 minutes** après l'entrée (choix avant évaluation pour une impulsion M15). Si aucune barrière touchée, sortie au CLOSE de la douzième M5. Aucun autre horizon ni variation de stop/target ne sera balayé dans V17.
- **Qualification :** exiger les 12 M5 contiguës, finissant **strictement avant** 2026-07-01 00 UTC, sans données de juillet–septembre 2026 ; sans imputation des trous ni prix d'exécution artificiel. Candidats exclus explicitement et réconciliés.
- **Coûts hypothétiques aller-retour :** exactement 2, 5 et 10 bp déduits des rendements bruts relatifs à l'open hypothétique. Funding réel, spread variable, slippage, modèle de remplissage, marge, coût capital/liquidation, exécution effective ne sont PAS simulés. Ces frais peuvent être trop optimistes pour des impulsions volatiles.
- **Aucun portefeuille capitalisé** : valeurs par signal uniquement, chevauchements entre actifs possibles. Ne pas appeler somme des bp un PnL de portefeuille.
- Comparateur `NO_TRADE` = 0 bp ; V14 reste un résultat exploratoire avec architectures/timeframes et freins non identiques (comparaison illustrative seulement).

## Sorties et gardes-fous

Nouveaux JSON et CSV des événements, hors de P0, jamais écrasés. Audit CSV V16 de référence non modifié. JSON : SHA-256 du fichier CSV V16, fingerprint P0, règles gelées, exclusions par symbole/sens, moyennes brute/net 2/5/10 bp, médiane, proportion positive net, distribution raisons d'exit, nombre de signaux V16 et de fenêtres admissibles, agrégats par actif et direction. CSV : événements comptabilisés, entrée/sortie et rendements hypothétiques, statut `HISTORICAL_EXPLORED_NOT_INDEPENDENT`. **Publier tous les groupes**, même si négatifs ; aucune sélection de ETH/SOL ni des directions après lecture.

### Décision

- Si le coût plancher 2 bp reste négatif : A06 configuration fixée **non convaincante** ; ne pas optimiser 0,20 / 1,5 / 0,35 ou horizon sur les mêmes rendements.
- Si 2 bp positif mais 5 bp négatif : signal trop sensible aux frais, pas de live ni ML.
- Si 5 bp positif : effectuer des diagnostics de stabilité et de frictions réalistes. **Pas de claim de rentabilité** tant qu'un nouvel échantillon indépendant n'est pas validé, et après correction des essais multiples.
- Quelle que soit l'issue, ne pas entraîner un sélecteur « meilleur trader » en lui révélant le PnL futur. Garder NO_TRADE.
- Funding/basis demeure une **piste économique séparée** : la commande V16 `--probe-funding-history` peut vérifier un échantillon public, mais ne construit ni un historique complet ni un backtest d'arbitrage.

**Statut : V17_RESEARCH_ONLY_NOT_VALIDATED.**
