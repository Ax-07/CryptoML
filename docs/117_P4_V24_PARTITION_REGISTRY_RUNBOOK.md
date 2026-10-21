# P4 V24 — Registre 84 partitions et reprise progressive

L'opérateur a confirmé V23 : **7 tests ciblés et suite complète passés**, quatre archives ETHUSDT juin 2025 vérifiées ; 8 640 M5 et 90 événements funding. V21 avait validé BTCUSDT juin 2025. **V24 n'a pas encore été exécuté sur le PC Windows.**

Lire [116_P4_V24_IMMUTABLE_PARTITION_REGISTRY_PROTOCOL.md](116_P4_V24_IMMUTABLE_PARTITION_REGISTRY_PROTOCOL.md).

## Installer et tester V24

\`\`\`powershell
cd E:\cryptoML-verified-partitions-v23
git fetch origin
git worktree add E:\cryptoML-partition-registry-v24 origin/research/funding-basis-partition-registry-v24
cd E:\cryptoML-partition-registry-v24\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_partition_registry_v24.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**STOP au moindre échec.** Le mode audit ne télécharge rien ; il vérifie réellement les manifestes, ZIP+CHECKSUM, SHA des Parquets, contenu M5, événements funding et identité P0.

## Audit initial strictement hors réseau

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_partition_registry_v24 \`
  --mode audit \`
  --data-root "E:\cryptoML\data" \`
  --v21-directory "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable" \`
  --v23-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" \`
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v24_registry_initial_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v24_registry_initial_v1.csv"
\`\`\`

Résultat **espéré mais non présumé** : \`VERIFIED_V21=1\`, \`VERIFIED_V23=1\`, \`MISSING=82\`, \`FAILED_INTEGRITY=0\`.

Si une partition obtient \`FAILED_INTEGRITY\`, transmettre le registre, **ne pas ingérer**.

## Premier lot autorisé : seulement BTCUSDT mai 2025

Après réussite de l'audit, créer un second rapport dans des **noms jamais utilisés** :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_partition_registry_v24 \`
  --mode ingest --permit-downloads \`
  --symbol BTCUSDT --start-month 2025-05 --max-new-partitions 1 \`
  --data-root "E:\cryptoML\data" \`
  --v21-directory "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable" \`
  --v23-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" \`
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv" \`
  --output-json "E:\cryptoML-walkforward-results\p4_v24_registry_after_one_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v24_registry_after_one_v1.csv"
\`\`\`

**ATTENTION :** \`--start-month\` est un curseur inclusif et l'outil sélectionne la première partition **manquante** de cet actif à partir de cette date. Avec \`BTCUSDT 2025-05\` et max 1, il tentera mai 2025 et ne retéléchargera pas le pilote BTC juin 2025. Une partition existante n'est jamais écrasée.

Si une archive ne respecte pas les contrôles, le téléchargement n'est pas validé ; le programme s'arrête, sans modifier P0 ni publier de partition invalide. Des sorties V24 déjà créées ne sont jamais remplacées.

**Aucune commande ici n'ingère automatiquement les 82 partitions restantes.** Aucun calcul de basis, funding carry, backtest, ML, ordre ou revendication OOS n'est réalisé.
