# P4 V8 — Procédure de comparaison économique mécanique

**Lire d'abord** [86_P4_V8_FROZEN_LONG_SHORT_ECONOMICS_PROTOCOL.md](86_P4_V8_FROZEN_LONG_SHORT_ECONOMICS_PROTOCOL.md). Les résultats V8 restent **hypothétiques**, sur les données historiques déjà inspectées. Aucune validation OOS, aucun ordre réel. Les financements et frictions propres aux perpétuels ne sont pas inclus au-delà des coûts fixes théoriques.

## Windows

\`\`\`powershell
cd E:\cryptoML-market-structure-v7
git fetch origin
git worktree add E:\cryptoML-long-short-economics-v8 origin/research/long-short-economics-audit-v8
cd E:\cryptoML-long-short-economics-v8\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_long_short_economics_audit_v8.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

# UNIQUEMENT si les deux tests sont verts
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.long_short_economics_audit_v8 \`
  --data-root "E:\cryptoML\data" \`
  --reference-json "E:\cryptoML-walkforward-results\p4_v6_market_structure_census_v1.json" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v8_symmetric_economics_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v8_symmetric_economics_v1.csv"
\`\`\`

Fichiers de sortie **nouveaux** et extérieurs à P0. N'utilisez pas un nom déjà existant.

## Arrêts stricts

Le programme doit échouer avant publication finale si :
- la signature des données, les états ou les effectifs de candidats diffèrent du V6 gelé ;
- le moindre markout LONG mature ne correspond pas **événement pour événement** aux labels V1 immuables (signal, confirmation B, horloges t/t+1/t+2, prix, ATR, raison de sortie, net) ;
- un bloc de six bougies de sortie est incomplet, manque le seuil de maturité ou croise juillet 2026 ;
- les résultats ALL ne sont pas l'union disjointe des sous-groupes ALIGNED et REJECTED.

## Lecture

- Coûts 2/5/10 bp notionnels complets par **trade**, et non frais réellement observés.
- \`net_5bps_per_trade\` = moyenne sur les trade retenus ; \`net_5bps_per_opportunity\` = somme des markouts retenus divisée par toutes les opportunités maturées du **même sens** (pour des comparaisons équitables de la sélection). Ce dernier nombre n'est pas un PnL de portefeuille.
- A = tout candidat ; B = confirmation \`t+1\` dans le sens de la position.
- ALL = tous, ALIGNED = M15 et H1 concordants à \`t\`, REJECTED = complément.
- Des LONG et SHORT peuvent se chevaucher sur le même actif ; leurs montants cumulés ne sont pas un compte de capital. L'audit compte les fenêtres qui se chevauchent.
- \`RANGE\` inclut les tendances insuffisamment confirmées, les transitions et les désaccords. Ne pas optimiser le seuil de ER12, l'écart EMA/ATR ni les durées après lecture de ce rapport.
- La comparaison D = NO_TRADE (0 markout) reste admissible conceptuellement ; les anciens candidats et les nouveaux SHORT ne deviennent **pas** une stratégie autorisée pour du trading en réel.

**Envoyer les sorties des tests et du recensement économique avant toute décision.**
