# P4 V6 — Clôture du recensement LONG/SHORT et contexte M15+H1 (2026-10-20)

**Statut : `UNLABELED_CENSUS_COMPLETE`. Aucune stratégie gagnante démontrée. Pas d'ordre ni de test de rendement V6.**

## Provenance et périmètre

Résultats déclarés dans une sortie PowerShell utilisateur, **non directement vérifiés** depuis les fichiers locaux JSON/CSV. Tests `tests/test_market_structure_v6.py` : 6/6 réussis ; `pytest -q` : aucun test en échec ; seuls avertissements de dépréciation Starlette/httpx et LightGBM. Le correctif `fb86eeb9de5a7df53fe709de038d7ffcef9aa45d` aligne les index Pandas des comparaisons H1/M15 sans modifier le code de stratégie.

Code : `backend/mlpipeline/market_structure_v6.py` et `market_structure_census_v6.py`. Fenêtre oct. 2024 – juin 2026, quatre actifs BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT ; Parquets P0 vérifiés SHA-256 ; ADA et juillet–septembre 2026 exclus.

## Décompte fixed-protocol

| Indicateur | Compte |
| --- | ---: |
| Bougies M5 | 734 972 |
| LONG raw (avant cooldown) | 27 547 |
| SHORT raw (avant cooldown) | 26 349 |
| LONG signaux après cooldown | 21 583 |
| SHORT signaux après cooldown | 20 674 |
| **Tous candidats après cooldown** | **42 257** |
| LONG avec contexte M15=H1=UP | 2 202 |
| SHORT avec contexte M15=H1=DOWN | 2 133 |
| **Candidats alignés (sans aucune simulation de prise de position)** | **4 335, soit ~10,26 %** |
| Autres candidats | 37 922 |

| Contexte global M15/H1 | Bougies | % |
| --- | ---: | ---: |
| UP | 32 670 | 4,45 |
| DOWN | 31 668 | 4,31 |
| RANGE (classe résiduelle) | 669 246 | 91,06 |
| UNKNOWN | 1 388 | 0,19 |
| **Total** | **734 972** | **100** |

La correspondance **exacte** du compte LONG V6 = 21 583 avec le nombre de signaux historiques LONG du détecteur V1 soutient une cohérence de nombre (pas une démonstration d'identité de tous les horodatages). Par crypto, les taux exacts et mois exacts figurent dans les sorties CSV/JSON locales non transmises.

## Restrictions d'interprétation

1. **La catégorie `RANGE` ne signifie pas « vraie consolidation horizontale ».** Elle rassemble marché en range, transitions, divergences M15/H1 et tendances ne satisfaisant pas les seuils très stricts (ER12 >= 0,35, spread EMA ATR >= 0,35 et pentes cohérentes). Le classificateur est une construction descriptive, pas une étiquette-vérité.
2. Le taux ~10,26 % est une **couverture** des signaux alignés sur les deux timeframes, **pas** un taux de réussite ni une performance. Un signal filtré n'implique aucun avantage.
3. LONG vs SHORT presque équilibré en nombre ; les comportements bruts et les coûts des SHORT sont **inconnus**. Ne pas confondre le fait que les données proviennent de contrats perpétuels avec l'exécution d'ordres.
4. Ce sont encore des périodes historiques déjà analysées pour les LONG. L'étude descriptive n'établit pas un OOS indépendant et ne légitime pas une optimisation des paramètres.

## Décision et progression contrôlée

- **Ne retoucher aucun seuil M15/H1** sur la base de la proportion de RANGE, d'un exemple ETH déjà inspecté, ni des résultats de mai–juin. 
- Prochaine vérification utile : **audit graphique descriptif** d'exemples temporels fixés à l'avance (dont ETH 2026-06-06 17:00 UTC déjà connu comme cas d'étude), affichant la structure M5, l'état M15/H1 **connu à t**, et l'étiquette globale. Les captures inspectées ne sont pas des preuves de profit.
- Avant tout calcul de rendement SHORT ou du filtre MTF, **geler un protocole séparé** (symétrie exacte des fills et barrières, chronologie t/t+1/t+2, priorités de sortie, coûts 2/5/10 bp, comparaison avec non-trade et fréquence par opportunité, traitements gaps et funding des perpétuels, séquences de commandes/tests). Ne pas inférer une rentabilité à partir du seul census.
- Confirmation possible seulement sur nouvelles prédictions futures, verrouillées avant observation.

**Statut final : `V6_UNLABELED_CONTEXT_DISCOVERY_REPORTED`.**
