# P4 V7 — Bilan visuel de l'analyse M5/M15/H1 (2026-10-20)

**Statut : audit descriptif terminé. Le signal ETH du 6 juin est écarté par le contexte V6, mais il n'existe toujours aucune validation économique d'une stratégie V6.**

## Source et intégrité

Résultats de PowerShell communiqués par l'opérateur : **5/5 tests V7 réussis**, suite complète sans échec (avertissements de dépréciation Starlette/httpx et LightGBM uniquement). Les 4 actifs ont été calculés, comparés au recensement V6 gelé, et les cinq cas ont produit des graphiques M5/M15/H1. L'opérateur a transmis le HTML exporté (environ 695 Ko), avec **15 panneaux SVG** ; les descriptions chiffrées ci-dessous ont été recoupées entre les logs et cet HTML. Ce n'est pas une relecture autonome du P0 sur le poste local. Les bougies viennent de l'archive Binance et les signaux sont issus des calculs historiques : **aucun ordre exécuté**.

Aucun rendement, prix d'entrée/sortie futurs, recalibrage de seuil, modification ML ou période réservée (ADA, juillet–septembre 2026) n'est intervenu.

## Cas ETH déjà inspecté — résultat

- **ETHUSDT LONG M5, signal clôturé le 2026-06-06 à 17:00 UTC**.
- M15 clôturée à **17:00 UTC** : **RANGE**, ER12 = **0.14376273488793068**, (EMA9−EMA26)/ATR14 = **−0.11245570530315965**.
- H1 clôturée à **17:00 UTC** : **RANGE**, ER12 = **0.2719731384554619**, (EMA9−EMA26)/ATR14 = **−0.8520676637703658**.
- Le seuil ER12 fixé auparavant est 0.35 pour classer UP ou DOWN. Aucune des deux unités ne donne UP ; la classe globale est RANGE, donc **LONG non aligné et écarté** par la politique descriptive d'alignement M15/H1.
- **Précision importante** : bien que classé RANGE par V6, le contexte H1 présente un écart EMA9−EMA26 négatif important, compatible avec un mouvement baissier et non nécessairement une consolidation latérale. RANGE veut dire « non classé dans les deux tendances strictes », incluant les transitions et désaccords. La catégorie ne décrit pas à elle seule la forme visuelle réelle du marché.
- Cette position ETH est une observation déjà inspectée après connaissance de sa perte historique en V1/C0. **Le filtre qui aurait évité ce perdant n'est pas prouvé profitable** sans l'exhaustivité des autres signaux retenus/refusés et des coûts.

## Quatre exemples choisis avant lecture des rendements

| Classe demandée (premier cas chronologique de mai) | Signal UTC | M15 ER / spread ATR | H1 ER / spread ATR | V6 global |
| --- | --- | --- | --- | --- |
| LONG avec UP concordant, BTCUSDT | 2026-05-01 04:00 | +0.652429 / +1.259544 | +0.371396 / +0.593389 | UP |
| SHORT avec DOWN concordant, ETHUSDT | 2026-05-07 01:30 | +0.356920 / −1.026694 | +0.746275 / −0.726511 (H1 clôturée 01:00) | DOWN |
| LONG classe RANGE, XRPUSDT | 2026-05-01 00:30 | +0.076923 / −0.092803 | +0.095745 / −0.339688 (H1 clôturée 00:00) | RANGE |
| SHORT classe RANGE, ETHUSDT | 2026-05-01 07:00 | +0.215810 / **+0.941327** | +0.310698 / **+0.487487** | RANGE |

Les deux exemples « alignés » contiennent un indicateur ER relativement proche du seuil 0.35 (BTC H1 0.371 ; ETH SHORT M15 0.357). Les deux cas RANGE ne constituent pas nécessairement des marchés plats ; le SHORT ETH du 1er mai possède notamment des écarts EMA/ATR **positifs sur M15 et H1**, opposés au sens SHORT. Des seuils figés produisent des classes utiles au recensement mais leurs frontières ne constituent pas une garantie de bonne classification.

## Conclusion et suite disciplinée

1. **La remarque humaine initiale était pertinente :** la seule EMA9/EMA26 M5 peut générer des LONG dans un contexte supérieur incompatible ; l'exemple ETH est confirmé temporellement sans utiliser de données futures.
2. Les classes V6 sont **techniquement cohérentes pour ces cinq cas**, mais il ne s'agit ni d'une validation générale du classificateur ni d'une preuve de rendement après coût. Les 91 % « RANGE » du recensement V6 doivent toujours être interprétés comme **classe résiduelle**, pas comme 91 % de latéralité.
3. **Aucune optimisation a posteriori** des seuils ER, spread ATR, unité de temps ou barrières ne doit provenir de ces graphiques.
4. Étape éventuelle suivante (une autre branche et protocole gelé avant lecture des résultats) : comparer en descriptif les candidats LONG et SHORT, alignés/non alignés, avec fills symétriques et frictions perpétuelles prudentes, et la référence non-trade. L'historique réutilisé **n'est pas un OOS indépendant** ; seule une validation prospective verrouillée pourrait établir un avantage exploitable.

**Décision V7 : `VISUAL_CONTEXT_AUDIT_COMPLETE_NO_ECONOMIC_CLAIM`.**
