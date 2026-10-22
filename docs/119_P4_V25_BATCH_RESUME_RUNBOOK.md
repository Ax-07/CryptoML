# CryptoML V25 — Plan hors réseau et micro-lot de deux partitions

V24 a été validé par l'opérateur : **BTCUSDT 2024-10**, **BTCUSDT 2025-05**, **BTCUSDT 2025-06** (V21) et **ETHUSDT 2025-06**. Registre : **4/84 vérifiées, 80 manquantes, zéro anomalie**.

Lire le [protocole gelé V25](118_P4_V25_BATCH_RESUME_FROZEN_PROTOCOL.md). V25 est **nouveau** : tests et exécution sur Windows non encore confirmés. Aucun rendement B01/B02, aucun ordre ni entraînement ML.

## Installation / tests (arrêt si échec)

\`\`\`powershell
cd E:\cryptoML-partition-registry-v24
git fetch origin
git worktree add E:\cryptoML-batch-resume-v25 origin/research/funding-basis-batch-resume-v25
cd E:\cryptoML-batch-resume-v25\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_batch_resume_v25.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

## Phase A : planifier, **aucun réseau**

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_batch_resume_v25 \`
  --mode plan \`
  --data-root "E:\cryptoML\data" \`
  --v21-directory "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable" \`
  --v23-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" \`
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv" \`
  --output-plan "E:\cryptoML-walkforward-results\p4_v25_microbatch_plan_v1.json"
\`\`\`

**Plan attendu :** \`initial_verified=4\`, \`initial_missing=80\` et cibles **ETHUSDT 2024-10** et **SOLUSDT 2024-10**. Ces valeurs doivent être confirmées en console ; ne pas lancer les téléchargements en cas de divergence ou de \`FAILED_INTEGRITY\`.

## Phase B : autoriser *explicitement* le micro-lot

**Ne lancer cette phase qu'après avoir examiné la sortie de la phase A.** Téléchargements publics par V23 : au plus 2 partitions, donc jusqu'à huit archives ZIP et huit CHECKSUM, traitement séquentiel.

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_batch_resume_v25 \`
  --mode execute --permit-downloads \`
  --plan-json "E:\cryptoML-walkforward-results\p4_v25_microbatch_plan_v1.json" \`
  --journal-jsonl "E:\cryptoML-walkforward-results\p4_v25_microbatch_journal_v1.jsonl" \`
  --summary-json "E:\cryptoML-walkforward-results\p4_v25_microbatch_summary_v1.json"
\`\`\`

La commande réaudite les 84 partitions **avant tout GET**, valide l'identité du rapport V22, journalise chaque événement sur disque, ignore les cibles déjà vérifiées et s'arrête à la première erreur. **Le plan reste limité à ces deux couples**, sans sélection dynamique ni boucle sur 80 partitions.

**Reprise :** en cas d'interruption, conserver le premier journal et le JSON plan, puis exécuter le même plan avec des **nouveaux chemins** de journal et de bilan (\`..._v2.jsonl\` / \`..._v2.json\`). La nouvelle exécution réauditera le stockage et sautera les partitions déjà vérifiées.

Si les deux archives sont réellement conformes, résultat visé : **6 verified / 78 missing**, non garanti avant exécution. Une erreur doit être examinée à partir du journal et des rapports, sans contourner les validations ni modifier P0.
