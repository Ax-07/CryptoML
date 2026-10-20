# P4 V9 — protocole prospectif : cassure de structure et retest M15/M5 sous régime H1

**STATUT : HYPOTHÈSE NOUVELLE, NON ÉVALUÉE.** Figée après la clôture V8. Cette hypothèse ne réutilise PAS le premier toucher EMA9 M5, dont le V8 n'a trouvé aucun avantage net aux coûts étudiés. Ne jamais requalifier le passé oct. 2024–sep. 2026 comme échantillon indépendant, ni traiter un signal rétroactif comme prédiction prospective.

## 1. Stratégie proposée, règles fixes sans calibration historique

Marché : Binance USD-M perpetual, 4 actifs BTCUSDT / ETHUSDT / SOLUSDT / XRPUSDT. Positionnement long ou short **papier exclusivement**. Un signal n'est pas un ordre.

Contexte H1 : reprendre tel quel la classification causale H1 V6 (EMA9/26, ER12, ATR14, pentes, coupures sur trous) ; LONG seulement si `UP`, SHORT seulement si `DOWN`. Ce filtre préexistant est une hypothèse stricte, pas une preuve de pertinence.

Structure M15 : dernier plus haut et dernier plus bas des **12 chandelles M15 précédemment clôturées**, **en excluant** la M15 qui se clôture à l'horodatage du signal. Une bougie M15 doit contenir exactement trois M5 consécutives. 12 bougies M15 = 3 heures. La zone doit être supérieure à zéro ; ne pas interpoler les trous ni les bougies en formation.

Cassure au close M5 `t` : après clôture M5, vérifier `close(t)>high(M15[12 précédentes]) +0.1 ATR14(M5,t)` (LONG) ou `close(t)<low(M15[12 précédentes])−0.1 ATR14(M5,t)` (SHORT). Le niveau M15 est gelé au moment du signal. Exiger que la clôture M5 précédente soit du côté **non cassé** pour éviter des entrées sur cassures anciennes.

Retest : parmi les **6 bougies M5 suivantes** (30 min) sans trou, un LONG doit toucher la zone `[niveau−0.1 ATR, niveau+0.1 ATR]` via LOW, puis **clôturer au-dessus du niveau** sur une bougie verte (`close>open`). SHORT : miroir HIGH de la zone, clôture sous le niveau sur bougie rouge. Une clôture M5 au-delà de `niveau−0.75 ATR` (LONG) / `niveau+0.75 ATR` (SHORT) annule immédiatement la recherche. **Pas de double candidature sur un même actif dans les 8 barres M5 suivant la cassure.**

Signal de décision : à la **clôture du retest confirmé**, jamais à la cassure initiale ; aucun résultat futur ni bar intrabar n'est utilisé dans ce signal. Une absence de retest, une H1 non tendancielle, un trou de données, ou une fenêtre incomplète = abstention. Si H1 a changé de sens entre cassure et clôture de retest : abstention. Ce sont des états, pas des transactions.

## 2. Validation indépendante : calendrier et provenance

Le code peut consulter des bougies antérieures **pour échauffement uniquement**, sans compter leurs performances. Pas de labels/retours V9 dans ce module. Les premiers signaux admissibles sont ceux dont la décision se clôture **au moins au début du 22 octobre 2026 UTC** et qui sont apparus après que les règles ont été figées sur GitHub. Une exécution ultérieure doit authentifier :
- l'heure de verrouillage du commit GitHub, le SHA exact, et le fait que les partitions de bougies utilisées ont été **ingérées avant la publication du signal**, sinon ce n'est qu'un replay historique ;
- pour le scanner hors-ligne manuel livré ici, la sortie porte toujours `OFFLINE_REPLAY_NOT_LIVE_ATTESTED`, même si les bougies sont postérieures au seuil chronologique ;
- journaux immuables/append-only et horodatage au moment de la clôture (pas reconstruits plusieurs jours plus tard) seront nécessaires avant toute prétention de preuve prospective.
- L'historique décembre 2025–juin 2026 et le holdout déjà inspecté juillet–septembre 2026 sont interdits à l'évaluation V9 (mais peuvent servir à l'initialisation avant le premier signal réellement nouveau ; le scanner ici accepte une fenêtre de bougies antérieures pour EMA H1).
- Le replay manuel **n'est pas une épreuve prospective certifiée**, car la connaissance préalable des bougies au moment d'exécution peut produire une illusion de prospective.

## 3. Risque et décision économique — NON implémentés ici

Aucun ordre ni exécution. Les niveaux TP, SL, taille, levier, funding, financement, spread, slippage, liquidation et politique des positions concurrentes ne sont PAS choisis par ce code. Toute proposition d'évaluation économique future devra figer séparément ces conventions, des frais réalistes et une référence zéro trade, **avant de révéler les résultats**. Pas de garantie que cette hypothèse produit une opportunité ni un avantage.

## 4. Gardes techniques

- M5 finales, UTC, prix OHLC valides, 4 symboles stricts ; pas de lissage des trous.
- Niveau M15 strictement antérieur au signal de cassure ; H1 connue au close de cassure et au close du retest ; le signal de décision n'existe qu'à la clôture du retest.
- Les tests font volontairement varier les bougies **futures** et vérifient que le passé reste immuable, y compris le niveau gelé, les states et la disponibilité.
- Ne modifier aucun seuil, période ou actif suite à l'absence de signaux. Figer d'abord les données source et leur horodatage.

**Décision V9 : `PAPER_SIGNAL_PROTOCOL_ONLY`; en attente d'observations indépendantes et de tests locaux.**
