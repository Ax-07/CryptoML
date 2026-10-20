# P4 V4 — Clôture de l'audit économique A/B sur 21 mois (2026-10-20)

**Décision : aucune stratégie mécanique A/B éligible, aucun nouvel entraînement ou réglage autorisé sur ces diagnostics.** Toute observation est **exploratoire sur données historiques déjà utilisées**, jamais une validation indépendante.

## Provenance

Sortie PowerShell transmise par l'opérateur le 20 octobre 2026. Le rédacteur n'a pas directement vérifié les JSON/CSV locaux, leur identité SHA-256, ni tous les quantiles et résultats mensuels. Références : `mlpipeline.pullback_economics_audit_v4`, `docs/79_P4_V4_MECHANICAL_ECONOMICS_AUDIT_RUNBOOK.md`, et gel V3 `docs/78_P4_V3_RIDGE_EXPLORATORY_CLOSEOUT.md`.

Tests selon l'opérateur : 5/5 ciblés passés ; suite complète `pytest -q` sans aucun test en échec (avertissements de dépréciation uniquement). Données lues par le chargeur V3 : 84 Parquets P0 pré-juillet 2026 dont SHA-256 vérifiés, 4 actifs BTC/ETH/SOL/XRP, de 2024-10 à 2026-06. Aucun ADA, juillet–septembre ni P3. **Hypothèses de prix et frais non exécutables** ; aucune tenue de portefeuille, spreads, slippage ou liquidité réels.

## Résultats exhaustifs sur le périmètre fixe

| Politique | Trades | Brut moyen bp/trade | Net à 2 bp/trade | Net à 5 bp/trade | Net à 10 bp/trade |
| --- | ---: | ---: | ---: | ---: | ---: |
| A : tous signaux V1 mûrs | 21 582 | +0.3090940915455281 | −1.690905908454472 | −4.690905908454472 | −9.690905908454472 |
| B : sous-ensemble confirmé t+1 | 9 238 | −0.11880050537182348 | −2.1188005053718235 | −5.118800505371824 | −10.118800505371824 |

Les sensibilités à 2/10 bp sont une **décomposition arithmétique** des résultats bruts, non des tests avec un fill différent. L'observation B−A est −0.42789459691735154 bp/trade brut, comparaison descriptive de deux ensembles de tailles différentes. Comme B est un sous-ensemble de A, son complément de 12 344 signaux a par bilan comptable un brut moyen d'environ +0.6293217556999751 bp/trade, toujours inférieur même à 2 bp de coûts supposés. Ce n'est pas une recommandation de supprimer/inverser la règle B.

## Motifs de sortie fixes

| Motif | A | B |
| --- | ---: | ---: |
| gap_stop | 1 | 1 |
| gap_target_capped | 0 | 0 |
| both_hit_stop_first | 52 | 33 |
| stop | 10 199 | 4 445 |
| target | 6 198 | 2 686 |
| timeout | 5 132 | 2 073 |
| **Total** | **21 582** | **9 238** |

Sur B, les arrêts par stop standard représentent environ 48.12 %, les objectifs environ 29.08 %, les timeouts environ 22.44 %, et 34 autres sorties (33 ambiguës stop-prioritaire + 1 gap_stop). Les fréquences seules ne démontrent ni problème de stop ni solution de changement de barrières.

## Limites et décision

- **A** possède un markout brut positif de +0.31 bp/trade mais **pas** un bénéfice net pour les trois frais supposés (2/5/10 bp).
- **B** est brut négatif ; sa confirmation ne produit pas de sélection profitable selon cette géométrie.
- Ces données n'identifient **pas** si les pertes découlent de la qualité d'entrée, de l'asymétrie stop/target, de la latence d'exécution, du spread, de la dépendance de régime ou de fortes queues de distribution. Le JSON local contient quantiles par cohorte et les tables mensuelles ; ceux-ci n'ont pas été reçus dans la sortie PowerShell et restent à **consulter descriptivement, sans seuil optimisé**.
- Ne pas changer la fenêtre historique, les six bougies, les barrières ou les paramètres ML d'après ces résultats. Le zéro trade reste la seule référence avec net zéro certain dans ce calcul.
- Pour explorer une **nouvelle hypothèse indépendante** : définir d'abord sa mécanique, sa métrique, ses tests et son calendrier d'évaluation *avant* d'accéder à ses résultats. Preuve confirmatoire seulement avec des prédictions véritablement futures verrouillées avant observation.

**Statut V4 : `MECHANICAL_AB_NO_ELIGIBLE_EDGE` ; branche conservée comme diagnostic immuable.**
