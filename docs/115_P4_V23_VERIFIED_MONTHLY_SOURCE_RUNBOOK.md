# P4 V23 — Première ingestion nouvelle partition ETHUSDT juin 2025

Les 3 tests V22, la suite entière et les inventaires ont été **validés par l'opérateur** : 48/48 et 336/336 requêtes HEAD retournent 200. **HEAD ≠ intégrité des ZIP.** V21 BTCUSDT juin 2025 a déjà produit 8640 M5 synchronisées et 90 funding.

Lire le [protocole V23 gelé](114_P4_V23_VERIFIED_MONTHLY_SOURCE_PROTOCOL.md). **V23 n'est pas encore exécuté sur le PC de l'opérateur.**

## Installation / tests

\`\`\`powershell
cd E:\cryptoML-funding-coverage-v22
git fetch origin
git worktree add E:\cryptoML-verified-partitions-v23 origin/research/funding-basis-verified-partitions-v23
cd E:\cryptoML-verified-partitions-v23\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_verified_partition_v23.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**STOP si un test échoue.** Ne pas ingérer tant que les tests ne passent pas.

## Premier nouveau test réel : ETHUSDT juin 2025

Créer la racine de sortie **hors P0** une fois (s'assurer auparavant de ne pas utiliser un chemin déjà destiné à d'autres données).

\`\`\`powershell
New-Item -ItemType Directory -Force "E:\cryptoML-walkforward-results\p4_v23_verified_sources" | Out-Null
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_verified_partition_v23 \`
  --ingest-one-partition --symbol ETHUSDT --month 2025-06 \`
  --data-root "E:\cryptoML\data" \`
  --output-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" \`
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv"
\`\`\`

Le script vérifie **seulement** les quatre ZIP+CHECKSUM du couple ETHUSDT juin 2025, leurs schémas, la régularité des bougies et le P0 ETHUSDT juin 2025. Funding demeure un tableau d'événements réels indépendant de la timeline M5. Il n'impose PAS un nombre fixe de paiements par mois. La destination est \`p4_v23_verified_sources\symbol=ETHUSDT\month=2025-06\`, **jamais écrasée**. Les 4 ZIP et checksums sont conservés dans \`raw\`, plus \`markets_5m.parquet\`, \`funding_events.parquet\` et \`manifest.json\`.

Ne pas relancer la même partition sans avoir examiné son manifeste ; un dossier existant est expressément refusé. Pas de boucle 84 partitions avant vérification du pilote et d'un plan de disque/reprise. Si un test réel échoue, transmettre toute la traceback sans contourner les contrôles.

### Décision suivante

Après ce second actif validé, prévoir un registre global des partitions, checksum et disponibilité disque, et une orchestration contrôlée et reprenable pour les 84 combinaisons, sans confondre couverture URL V22 et données vérifiées. Aucune performance funding/carry, pas de backtest, pas de nouvelle OOS ni d'ordres réels à cette étape.
