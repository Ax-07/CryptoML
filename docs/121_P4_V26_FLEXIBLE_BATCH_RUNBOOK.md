# CryptoML V26 — Micro-lots paramétrables, plan signé et reprise

V25 : exécution réussie des deux partitions ETHUSDT et SOLUSDT octobre 2024, registre local à **6 / 84 vérifiées**, **78 manquantes**. Le nouveau V26 est publié mais **pas encore exécuté sur l'environnement Windows**.

Protocole gelé : [120_P4_V26_FLEXIBLE_BATCH_FROZEN_PROTOCOL.md](120_P4_V26_FLEXIBLE_BATCH_FROZEN_PROTOCOL.md).

## Installer et tester

\`\`\`powershell
cd E:\cryptoML-batch-resume-v25
git fetch origin
git worktree add E:\cryptoML-flexible-batches-v26 origin/research/funding-basis-flexible-microbatches-v26
cd E:\cryptoML-flexible-batches-v26\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_flexible_microbatch_v26.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**STOP au moindre test en échec.**

## Phase A — construire le plan HORS RÉSEAU

Choix préenregistré : **XRPUSDT 2024-10** + **ETHUSDT 2024-11**. Aucun signal ou rendement historique n'est regardé.

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_flexible_microbatch_v26 \`
  --mode plan \`
  --target XRPUSDT:2024-10 --target ETHUSDT:2024-11 \`
  --data-root "E:\cryptoML\data" \`
  --v21-directory "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable" \`
  --v23-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" \`
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv" \`
  --output-plan "E:\cryptoML-walkforward-results\p4_v26_microbatch_plan_v1.json"

Get-FileHash "E:\cryptoML-walkforward-results\p4_v26_microbatch_plan_v1.json" -Algorithm SHA256
\`\`\`

Attendu (à confirmer par la console) : **6 verified / 78 missing**, cibles XRP octobre 2024 et ETH novembre 2024. Conserver la valeur SHA256 retournée pour la phase d'exécution. Les fichiers de sortie sont toujours **nouveaux**.

## Phase B — téléchargement volontaire (après validation du plan)

Ne jamais inventer le SHA256. Le fournir en remplaçant \`<SHA256_AFFICHE_PAR_GET_FILEHASH>\` par **la valeur exacte** (64 caractères hex). Le processus relit le plan, compare son SHA, ré-audite 84 partitions et refuse toute anomalie, avant huit GET ZIP et huit GET CHECKSUM au maximum.

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_flexible_microbatch_v26 \`
  --mode execute --permit-downloads \`
  --plan-json "E:\cryptoML-walkforward-results\p4_v26_microbatch_plan_v1.json" \`
  --plan-sha256 "<SHA256_AFFICHE_PAR_GET_FILEHASH>" \`
  --journal-jsonl "E:\cryptoML-walkforward-results\p4_v26_microbatch_journal_v1.jsonl" \`
  --summary-json "E:\cryptoML-walkforward-results\p4_v26_microbatch_summary_v1.json"
\`\`\`

En cas de panne, **ne pas effacer** la partition validée ou le journal. Conserver le même plan et sa référence SHA, relancer avec de **nouveaux noms** de journal/bilan (par exemple \`_v2\`). L'outil réaudite le registre et ignore les cibles déjà conformes. Une cible incomplète bloque la reprise jusqu'à diagnostic manuel.

Succès espéré : 8/84 partitions vérifiées et 76 manquantes, **non garanti** avant tests réels. Ne jamais ingérer automatiquement les 78 couples restants. Pas de backtest funding/basis, pas de ML, pas d'ordre.
