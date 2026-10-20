# P4 V8 — Clôture audit économique LONG / SHORT (rapporté le 21 octobre 2026)

**Décision : `NO_ELIGIBLE_EDGE` — résultats négatifs sur tous les groupes agrégés même à 2 bps, aucune stratégie déployable validée.** Aucun ordre exécuté, aucun réentraînement, aucune optimisation.

## Provenance et contrôles

Source : sortie PowerShell fournie par l'opérateur Windows, exécutant `backend/mlpipeline/long_short_economics_audit_v8.py` sur les Parquets P0 locaux. Les fichiers finaux JSON et CSV n'ont pas été lus directement par l'auteur de cette clôture ; les chiffres suivants sont ceux imprimés par le programme. Tests V8 : 15/15 réussis. Suite complète : aucun échec, seulement des avertissements de dépréciation Starlette/httpx et LightGBM. Le contrôle de référence V6, les décomptes et la parité événement-par-événement des LONG V1 ont terminé sans exception. Le correctif préalable `096eb25fbf7289a178217d08873a39a29f503735` a corrigé le `.all()` manquant sur une comparaison de série, et un test événement synthétique forcé a été ajouté dans `4ba7fd0ee4110f89e03ddc31dd13bb55fdcd62fa`.

Univers : BTCUSDT, ETHUSDT, SOLUSDT et XRPUSDT, octobre 2024 à juin 2026 ; SHA-256 et qualité P0 via chargeur historique ; ADA et juillet–septembre 2026 non utilisés. Observations historiques déjà inspectées, **jamais OOS indépendant**.

## Réconciliation de l'univers

| Actif | Candidats | Maturés | Exclus | LONG maturés | SHORT maturés |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 10 627 | 10 627 | 0 | 5 480 | 5 147 |
| ETHUSDT | 10 504 | 10 504 | 0 | 5 432 | 5 072 |
| SOLUSDT | 10 550 | 10 549 | 1 | 5 367 | 5 182 |
| XRPUSDT | 10 576 | 10 576 | 0 | 5 303 | 5 273 |
| **Total** | **42 257** | **42 256** | **1** | **21 582** | **20 674** |

4 335 événements A alignés à la classe M15/H1 V6 (2 202 LONG + 2 133 SHORT). 37 921 rejetés sur l'univers maturé.

## Moyenne hypothétique NET PAR TRADE à 5 bps aller-retour

| Sens / cohorte | ALL n | ALL net5 moyen (bp) | ALIGNED n | ALIGNED net5 moyen (bp) | REJECTED n | REJECTED net5 moyen (bp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LONG A | 21 582 | −4.690906 | 2 202 | −4.873505 | 19 380 | −4.670159 |
| LONG B | 9 238 | −5.118801 | 898 | −5.078415 | 8 340 | −5.123149 |
| SHORT A | 20 674 | −4.337870 | 2 133 | −3.710292 | 18 541 | −4.410068 |
| SHORT B | 8 751 | −3.980110 | 881 | −3.833573 | 7 870 | −3.996514 |
| BOTH A | 42 256 | −4.518181 | 4 335 | −4.301156 | 37 921 | −4.542990 |
| BOTH B | 17 989 | −4.564869 | 1 779 | −4.461942 | 16 210 | −4.576165 |

A = tous les signaux, B = confirmation en `t+1`. ALIGNED est la cohérence strictement UP/UP pour LONG, DOWN/DOWN pour SHORT connue à `t`. La baisse des pertes *par opportunité* en réduisant le nombre de trades n'est **pas** en elle-même une preuve d'amélioration de la sélection ; référentiel zéro-trade disponible.

### Frictions 2 / 5 / 10 bps : cohorte BOTH B ALIGNED

- Brut moyen théorique = **+0.538058 bp/trade** (net5 +5).
- Net moyen à **2 bp** = **−1.461942 bp/trade**.
- Net moyen à **5 bp** = **−4.461942 bp/trade**.
- Net moyen à **10 bp** = **−9.461942 bp/trade**.
- Amélioration de sélection conditionnelle versus BOTH B ALL : **+0.102927 bp/trade** ; **pas suffisante pour absorber même les 2 bp les plus faibles**.
- B ALIGNED conserve 1 779 / 17 989 = **~9.89 %** des B mûris. L'abstention réduit l'exposition, mais NO_TRADE = zéro markout hypothétique et reste supérieur à toutes les moyennes nettes mesurées à 2, 5 et 10 bp.

### Différences directionnelles

- LONG A se dégrade avec la confirmation MTF : −4.69 → −4.87 bp/trade.
- SHORT A s'améliore légèrement : −4.34 → −3.71 bp/trade.
- SHORT B s'améliore légèrement : −3.98 → −3.83 bp/trade.
- Toutes restent déficitaires, **y compris après hypothèse de 2 bp** par trade.

## Restrictions et décision de recherche

Le bilan totalise des outcomes notionnels individuels sur des fenêtres qui peuvent se chevaucher ; **ce n'est pas le rendement d'un portefeuille**. Les stops, objectifs, entrées exactes au OPEN(t+2) restent simulés sur OHLC M5, priorité stop pour touches ambiguës ; les coûts 2/5/10 bps ne représentent pas une exécution mesurée et **n'incluent pas explicitement funding, slippage dynamique, spread, levier, liquidation, tailles, exposition simultanée ni mark-price**. Ces coûts peuvent se révéler plus élevés et amplifier les pertes.

**Décision : ne pas déployer cette stratégie. Ne pas entraîner de nouveau ML sur ce même signal en présumant une rentabilité ni ajuster les seuils ER, EMA/ATR, barrière ou horizon post hoc sur V8.** Les métriques historiques sont exploratoires et non indépendantes (ETH connu). Une nouvelle hypothèse de recherche exigerait pré-enregistrement séparé et une validation prospective verrouillée, avec des coûts d'exécution réalistes, avant prétention d'avantage économique.

Statut final : `V8_HISTORICAL_ECONOMICS_COMPLETE_NO_ELIGIBLE_EDGE`.
