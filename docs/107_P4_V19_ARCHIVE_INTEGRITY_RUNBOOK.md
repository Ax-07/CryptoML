# V19 — Vérification réelle des archives BTCUSDT de juin 2025

**Rappel V18 :** 12/12 HEAD 200 et 7 paiements historiques funding via REST. Le code V19 n'a pas encore été testé sur le PC Windows de l'opérateur.

## Installer

\`\`\`powershell
cd E:\cryptoML-funding-basis-v18
git fetch origin
git worktree add E:\cryptoML-funding-integrity-v19 origin/research/funding-basis-archive-integrity-v19
cd E:\cryptoML-funding-integrity-v19\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_archive_integrity_v19.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**STOP si le moindre test échoue.**

## Audit volontaire, borné

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_archive_integrity_v19 \`
  --inspect-public-archives \`
  --output-json "E:\cryptoML-walkforward-results\p4_v19_archive_integrity_btc_2025_06_v1.json"
\`\`\`

Le script télécharge au plus quatre ZIP **BTCUSDT juin 2025** (spot, perp, mark, funding) et quatre textes \`.CHECKSUM\`. Chaque ZIP est limité à 40 MB en téléchargement, 100 MB après décompression, puis SHA256 est comparé au manifeste CHECKSUM. Aucun fichier prix ou funding n'est conservé et aucune partition P0 modifiée. Il écrit un **nouveau** JSON de synthèse hors P0. Les données funding n'étant pas garanties de partager un schéma CSV unique, \`SCHEMA_REVIEW_REQUIRED\` est un résultat attendu, pas un succès de parsing.

Après la sonde, transmettre l'output PowerShell. Seule une validation de schémas et de couverture sur quatre symboles × 21 mois pourra autoriser un futur dataset B01/B02 ; les réponses de V19 n'attestent rien de tel. Aucun backtest, entraînement ni ordre réel.
