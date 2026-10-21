# P4 V11 — Pré-enregistrement : audit économique exploratoire de V9

**Statut : protocole gelé AVANT toute ouverture des rendements V11.** Le moteur de cassure/retest V9 est recopié à l'identique depuis la V10 (y compris le contrôle de date de cassure), dans un module `historical_research` distinct, à l'exception de sa borne de démarrage historique. Les tests doivent vérifier sa parité comportementale avec V9 pour les dates postérieures au 22 octobre 2026. Les fichiers du scanner V9 et de la collecte V10 restent inchangés.

## Question de recherche

Une cassure du haut/bas des 12 précédentes M15 complètes, suivie d'un retest M5 confirmé sous régime H1 strict V6, capte-t-elle un mouvement brut suffisant pour absorber les frais ? Cette question a été formulée après l'échec de la stratégie V8. Toute comparaison ici est descriptive, sur **l'historique déjà exploré**, et ne prouve pas une amélioration OOS. Ne pas optimiser les seuils ni choisir les périodes selon les résultats.

## Univers et protections

BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, 21 mois **octobre 2024 à juin 2026** ; chargeur P4 V3, 84 Parquets P0 dont SHA-256, qualité, horloges et finalité sont vérifiés. Les données ADA et juillet–septembre 2026 sont exclues. Aucun accès à V10, au carnet d'ordres, aux clés API ni aux ordres. Les premiers jours peuvent être seulement du warmup ; aucun signal est ajouté artificiellement. Les exclusions pour ATR nul, maturité, trous ou bougies manquantes doivent être recensées.

## Scanner V9 gelé

Paramètres **inchangés** : H1 `UP` pour LONG / `DOWN` pour SHORT ; M15 plus haut/bas des 12 dernières M15 complètes **excluant le bucket du signal** ; cassure au-delà du niveau de `0,10 × ATR14(M5)`, close précédent non cassé ; retest dans les 6 bougies M5 suivantes (zone ±`0,10 ATR`), clôture directionnelle et bougie verte/rouge ; invalidation à `0,75 ATR`, H1 inchangé, cooldown de 8 bougies. Les décisions sont observables à la **clôture du retest**, pas à celle de la cassure. Interdiction de consulter t+1 comme critère d'entrée.

## Outcomes mécaniques, fixés avant exploration

Entrée simulée **à l'OPEN de la première bougie M5 après la clôture du retest**, sans latence ni écart bid/ask : hypothèse idéale, pas ordre exécutable.

Barrières : ATR14(M5) gelé à la *cassure*, LONG stop `E−ATR`, target `E+1,5 ATR`; SHORT stop `E+ATR`, target `E−1,5 ATR`. Pas de trailing, pas de filtre supplémentaire. Priorité au stop si ambiguïté OHLC ; gap adverse à l'open pire ; gap favorable plafonné au target ; à défaut sortie au close de la dernière bougie. Réutilisation de la fonction mécanique V8 `_barrier` **sans modifications**.

Horizon **primaire : 12 bougies M5 (60 min)**, choisi a priori pour donner plus de temps au retest de structure que l'horizon V8 (30 min). Deux **sensibilités non sélectionnables** de 6 (30 min) et 24 (120 min) bougies. **Même cohorte d'événements maturés sur 24 bougies pour les trois horizons** : six résultats ne peuvent pas inclure des événements que 24 exclut. Aucun meilleur des trois horizons ne sera choisi comme « stratégie gagnante » a posteriori.

Frais supposés aller-retour **2 / 5 / 10 bp**, identiques V8, avec moyenne brute, net/trade, net/opportunité, win rates, motifs stop/target/timeout et comptes. Funding perpétuels, slippage, spread variable, maker/taker exact, latence, liquidation, levier, taille, mark price NON modélisés ; ces hypothèses peuvent surestimer les résultats. Zéro trade reste un comparateur explicite.

Diagnostics descriptifs prévus pour **les deux directions par symbole et par mois**, et agrégés pour chaque horizon. Ne publier **aucune** sélection de gagnant par mois/symbole/horizon ; résultats par segment destinés au diagnostic de stabilité, pas à une optimisation.

## Conditions pour poursuivre ou arrêter

1. Contrôles non-fuite temporelle, parité V9 sur signaux synthétiques futurs, gestion des trous, prix et frais, comptabilité des exclusions.
2. Si le primaire à 12 M5 est négatif au coût plancher de 2 bp : **hypothèse non convaincante, STOP**. Ne pas réoptimiser sur ce même jeu.
3. Même si le primaire est positif à 2/5/10 bp, il demeure **exploratoire**, non rentable démontré ; préenregistrer une validation temporellement indépendante avec spreads/funding réalistes. Si la performance est portée par un seul mois ou actif, aucune généralisation.
4. Comparaison V8 limitée aux indications de couverture et d'espérance, car les univers d'événements et les horizons sont différents ; aucune équivalence de backtest implicite.
5. Rapport de sortie JSON + CSV nouveaux hors P0. Aucun graphique/chiffre historique prédéfini ; aucun retour supposé avant exécution vérifiée.

**Décision initiale : V11_RESEARCH_NOT_VALIDATED ; aucun trading réel ni paper monitoring automatique.**
