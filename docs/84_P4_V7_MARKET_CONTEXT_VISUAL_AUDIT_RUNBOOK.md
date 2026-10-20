# P4 V7 — Audit visuel du contexte causal M5 / M15 / H1

**Aucun PnL, retour futur, trade exécuté, réentraînement ou optimisation.** Version strictement descriptive fondée sur les bougies M5 Binance P0 immuables (BTC/ETH/SOL/XRP d'oct. 2024 à juin 2026). Ni ADA, ni mois protégés juillet–septembre 2026. La référence des effectifs V6 est vérifiée avant tout rendu.

## Cas choisis AVANT le rendu

1. ETHUSDT 2026-06-06 17:00 UTC : cas **déjà observé**. La V7 montrera son état V6 au moment du signal, sans consulter la sortie perdante, qui ne sert que de contexte à la question humaine.
2. Premier candidat LONG **UP** et premier candidat SHORT **DOWN** chronologiquement sur les quatre actifs en mai 2026.
3. Premier candidat LONG **RANGE** et premier candidat SHORT **RANGE** chronologiquement sur les quatre actifs en mai 2026.
4. Si aucune observation d'une classe, afficher **« aucune observation »** sans remplacer par un exemple favorable.

Ces exemples sont sélectionnés par horodatage et étiquette V6, sans utiliser les profits. Leur inspection humaine ne constitue pas une validation économique.

## Graphiques et contrôles

- Trois séries réelles de chandeliers **M5, M15 et H1**, accompagnées des EMA9/EMA26 ; la fenêtre s'arrête **au signal M5** ou à la dernière bougie M15/H1 intégralement clôturée connue à cet instant. Pas de bougie post-signal.
- États UP/DOWN/RANGE/UNKNOWN, ER12 et écart EMA9−EMA26 normalisé ATR, tirés du même calculateur V6 et comparés strictement à son as-of par bougie. Toute incohérence déclenche une erreur.
- L'axe H1 donne le contexte avant le signal ; l'entrée simulée OPEN(t+2) n'est ni tracée ni incluse dans la classification.
- Catégorie **RANGE** : étiquette résiduelle incluant divergences M15/H1, transitions et tendances inférieures aux seuils. Elle ne prouve **pas** une consolidation horizontale.
- Les nombres V6 par actif et dans le total doivent concorder exactement avec **le JSON immuable** \`p4_v6_market_structure_census_v1.json\` et sa signature de jeu de données. L'analyse échoue si les nombres diffèrent.
- Deux sorties **nouvelles**, un HTML autonome avec SVG (sans matplotlib) et un JSON avec l'état exact. Les données d'origine restent en lecture seule.

## Windows PowerShell

\`\`\`powershell
cd E:\cryptoML-market-structure-v6
git fetch origin
git worktree add E:\cryptoML-market-structure-v7 origin/research/market-structure-visual-audit-v7
cd E:\cryptoML-market-structure-v7\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_market_structure_visual_audit_v7.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

# Lancer UNIQUEMENT si les deux vérifications sont vertes
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.market_structure_visual_audit_v7 \`
  --data-root "E:\cryptoML\data" \`
  --reference-json "E:\cryptoML-walkforward-results\p4_v6_market_structure_census_v1.json" \`
  --output-html "E:\cryptoML-walkforward-results\p4_v7_market_context_visual_audit_v1.html" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v7_market_context_visual_audit_v1.json"

Start-Process "E:\cryptoML-walkforward-results\p4_v7_market_context_visual_audit_v1.html"
\`\`\`

Les nouveaux tests sont synthétiques ; ils ne lisent pas P0 réel. Les résultats de marché réels n'ont pas été exécutés ou vérifiés par le rédacteur sur le poste Windows.

## Arrêt méthodologique

Regarder d'abord si le cas ETH était effectivement classé **RANGE**, **UP**, **DOWN**, ou **UNKNOWN** à la clôture du signal, puis regarder *pourquoi* sur chaque timeframe. Une classification « RANGE » pourra être une divergence entre timeframes et non un range visuel. Ne modifier **aucun** seuil ni décider d'une stratégie gagnante à partir des captures. Tout protocole d'issues LONG/SHORT devra être documenté et gelé séparément avant analyse des rendements, puis validé prospectivement.
