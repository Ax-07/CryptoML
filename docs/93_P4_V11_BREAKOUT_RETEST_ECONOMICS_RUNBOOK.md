# P4 V11 — Exécution de l'audit économique V9 (historique, pas OOS)

Protocole gelé dans [92_P4_V11_FROZEN_BREAKOUT_RETEST_ECONOMICS_PROTOCOL.md](92_P4_V11_FROZEN_BREAKOUT_RETEST_ECONOMICS_PROTOCOL.md).

La V10 reste suspendue : pas besoin de lancer le collecteur en temps réel, pas d'ordre ni de position papier. **V11 répond uniquement à la question de recherche : le signal cassure/retest capte-t-il plus que ses frais ?**

## Installation Windows PowerShell

\`\`\`powershell
cd E:\cryptoML-paper-observer-v10
git fetch origin
git worktree add E:\cryptoML-breakout-retest-economics-v11 origin/research/v9-breakout-retest-economics-v11
cd E:\cryptoML-breakout-retest-economics-v11\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_structure_breakout_retest_economics_v11.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

Si les deux séries de tests réussissent :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.structure_breakout_retest_economics_v11 \`
  --data-root "E:\cryptoML\data" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v11_breakout_retest_economics_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v11_breakout_retest_economics_v1.csv"
\`\`\`

Les fichiers de sortie sont **nouveaux**, hors P0 et non écrasés. Le chargeur lit uniquement quatre symboles et 21 mois octobre 2024 → juin 2026. Les classes H1 et la structure M15 sont reconstruites causales à partir des M5 finalisées. Le scan est potentiellement coûteux sur 21 mois : ne pas interrompre par erreur pendant les longs calculs, mais surveiller que les quatre symboles s'affichent.

## Interprétation et arrêt

Les résultats agrégés contiennent, pour chacune des 3 fenêtres fixées (6/12/24 M5), LONG / SHORT / combiné : décompte, moyenne brute et net après 2/5/10 bps, raisons de sortie et diagnostics par crypto / mois dans le CSV.

**Critère primaire : 12 M5, net après 2 bp.** Si cette moyenne est négative, considérer la proposition insuffisante et ne pas optimiser les seuils sur ces mêmes données. Le rendement brut moyen peut permettre de comprendre si le problème vient du signal, des objectifs, de l'horizon ou des coûts, mais ne justifie pas un nouveau réglage opportuniste. Si le résultat est positif, il reste exploratoire et exige ensuite une validation indépendante.

La somme des bps n'est **pas** un rendement de portefeuille : chevauchements possibles, aucune taille, capital, gestion des positions concurrentes, financement des perpétuels ou slippage dynamique.

Les anciens résultats V8 restent inchangés. Les tests V11 sont synthétiques ; ils ne prouvent pas la rentabilité.

**Envoyer les sorties de tests et d'audit pour analyse.**
