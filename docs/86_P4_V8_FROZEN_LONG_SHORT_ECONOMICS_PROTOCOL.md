# P4 V8 — Préenregistrement économique LONG / SHORT, contextes V6 gelés

**Statut initial : PROTOCOLE GELÉ AVANT OUVERTURE DES LABELS V8.** Ce protocole ne prétend à aucune performance prospective : le jeu 2024-10 à 2026-06 est déjà exposé aux recherches P4. Le cas ETH du 6 juin 2026 a été inspecté à l'avance ; ce cas n'est pas un holdout. Ne pas changer ces règles après avoir vu les résultats.

## Hypothèse testée et comparaisons imposées

Tester uniquement si la sélection de signaux cohérents avec la tendance stricte de V6 améliore, sur cet historique déjà vu, les hypothétiques résultats *par opportunité* des pullbacks LONG et SHORT. N'interpréter aucun score comme stratégie tradable sans validation prospective.

Univers immuable : BTCUSDT ETHUSDT SOLUSDT XRPUSDT, **octobre 2024 inclus à juin 2026 inclus**, 84 partitions Parquet historiques vérifiées P0 SHA-256 et qualité ; aucun ADA, ni juillet–septembre 2026, ni P3. Le JSON de recensement V6 sert de **contrôle strict d'identité** : fingerprint, totaux par actif et globaux pour les signaux et les classes.

Signaux :
- Cohorte **A** : tout premier touch EMA9 M5 sélectionné par l'algorithme fixe V6, cooldown de 8 bougies, en **LONG** ou **SHORT**.
- Cohorte **B** : sous-groupe A vérifiant à la clôture de `t+1` : LONG `close(t+1)>open(t+1)`, `close(t+1)>EMA9(t+1)`, `close(t+1)>close(t)` ; pour SHORT les trois inégalités strictes inversées. Aucune étiquette de retour impliquée dans B.
- **ALL** = la cohorte A ou B non filtrée ; **ALIGNED** = LONG uniquement si M15 et H1 V6 sont tous deux `UP` à la clôture de `t`, SHORT uniquement si tous deux `DOWN` ; **REJECTED** = complément strict dans chaque cohorte, incluant `RANGE`, `UNKNOWN` et sens non concordant. Ces classes sont gelées. Classe `RANGE` demeure résiduelle, pas preuve de latéralité.
- **NO_TRADE** = zéro ordre, net théorique nul. Aucun portefeuille ni politique de conflits d'exécution affirmée.

## Barrières et chronologie, figées

Signal connu au **close(t)** ; B devient connu au **close(t+1)** ; entrée hypothétique au **OPEN(t+2)** = à la même date que close(t+1) pour bougies M5 consécutives ; entrée exacte supposée à l'open, pas une garantie de fill. Dépassements de gaps/latences réels inconnus.

Pour les **six bougies M5** de la fenêtre `t+2…t+7`, soit maturité théorique `close(t)+35 minutes`, sans trous ni périodes protégées :
- LONG : stop `E−1 ATR14(t)`, target `E+1.5 ATR14(t)`; SHORT : stop `E+1 ATR14(t)`, target `E−1.5 ATR14(t)`.
- Stop prioritaire si stop et target touchés dans la même bougie (OHLC ne révèle pas l'ordre des touches).
- Gap **défavorable** au-delà du stop : fill à l'open réellement observé de la bougie (moins favorable) ; gap **favorable** au-delà du target : fill **plafonné** au niveau du target. Si aucune barrière atteinte : sortie au close de la sixième bougie. Pas d'inférence sur l'ordre intra-bougie.
- `gross_bps=10000 × direction × (exit−entry)/entry`, direction LONG=+1, SHORT=−1 ; `net_2bps`, `net_5bps`, `net_10bps` = brut moins coût hypothétique fixe **aller-retour par trade** de 2, 5, 10 bp notionnels.
- **Les coûts NE modélisent pas explicitement funding des perpétuels, spread variable, slippage réel, maker/taker exact, liquidation, levier et gestion des marges ; les résultats ne sont pas un PnL Binance réel.**
- Écarter tout signal sans 6 bougies consécutives, ATR fini >0, stop/target économique valide (>0), maturité strictement antérieure à `2026-07-01T00:00:00Z`. Publier les exclusions/immaturités en décomptes, ne jamais les imputer.
- Confirmation B ne peut utiliser que t+1, et la classe MTF est figée à t, non recalculée avec futur.

## Rapport et vérifications immuables

Rapport séparé par sens LONG/SHORT et global, par mois et symbole, pour **A et B**, et pour chaque sélection **ALL / ALIGNED / REJECTED**, **plus NO_TRADE**. Pour chaque comparaison :
- nombre d'opportunités éligibles (maturées) fixe au niveau A pour le sens considéré, trades, couverture, somme des bps notionnels, moyenne/trade et moyenne/opportunité aux trois coûts ;
- nombre de gains/pertes théoriques aux trois coûts, répartition stop/target/timeout/gap, médiane ;
- conservation `ALL=ALIGNED+REJECTED` pour nombre, somme brut/net, issues ; conserver B sous-ensemble de A ; périodisation mensuelle et agrégation actifs ;
- différence ALIGNED−ALL par opportunité et diagnostic du coût d'opportunité : pertes **et gains** évités/exclus ;
- référence de cohérence exhaustive LONG contre l'ancien outcome V1 pour chaque événement maturé, y compris prix/sortie/motif/gross ; tout écart bloque la publication.
- détecter les fenêtres de trade théorique recouvrantes en sens opposés sur un même symbole ; rapporter le nombre, sans agréger des bps comme rendement d'un portefeuille réalisable.

Interdictions : entraîner ML, optimiser seuils, tester des sous-groupes rétrospectifs pour choisir un « gagnant », utiliser des labels V7/V6, ouvrir ADA ou mois post juin, inventer frais réels, publier des gains de portefeuille. Les bps sommés servent de **diagnostic de couverture notionnelle à mise unitaire**, et non de compte de capital.

Les fichiers sont obligatoirement **nouveaux**, hors P0 : JSON descriptif et CSV par mois/sens/cohorte. S'il y a une erreur de contrôle de cohérence, ne pas produire de rapport final.

Décision suivant résultats : déclarer `NO_ELIGIBLE_EDGE` tant qu'aucune prédiction future verrouillée en amont n'a démontré, sur données indépendantes, un avantage après coûts et modèle d'exécution adéquat. Pas d'ordre réel.
