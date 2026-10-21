# CryptoML V22 — Inventaire de couverture funding / basis

**Point de départ confirmé par l'opérateur :** V21, 7 tests ciblés et suite complète réussis ; BTCUSDT juin 2025 : 8 640 M5 synchronisées et 90 événements funding, avec SHA256 et P0 contrôlés. V22 ne mesure pas de rentabilité.

[Protocole V22](112_P4_V22_ARCHIVE_COVERAGE_FROZEN_PROTOCOL.md)

## Installation et vérification

\`\`\`powershell
cd E:\cryptoML-funding-pilot-v21
git fetch origin
git worktree add E:\cryptoML-funding-coverage-v22 origin/research/funding-basis-archive-coverage-v22
cd E:\cryptoML-funding-coverage-v22\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_archive_coverage_v22.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Ne pas exécuter la sonde avant les tests.** Aucune exécution Windows V22 n'a encore été confirmée.

## Inventaire échantillon (48 requêtes HEAD, aucun ZIP)

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_archive_coverage_v22 \`
  --probe-public-archives --mode sample \`
  --output-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_sample_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_sample_v1.csv"
\`\`\`

## Inventaire complet seulement après réussite de l'échantillon (336 HEAD)

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_archive_coverage_v22 \`
  --probe-public-archives --mode full \`
  --output-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" \`
  --output-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv"
\`\`\`

Les chemins des quatre sorties doivent être **nouveaux**, en dehors de \`E:\cryptoML\data\` pour préserver P0. Le programme n'a pas besoin du P0 pour ce recensement. Quatre requêtes au maximum en parallèle, 12 s de timeout individuel, aucun ZIP ni CHECKSUM téléchargé, aucune clé API.

Résultats \`HEAD_OK_UNVERIFIED\` = HTTP 200 / source potentiellement accessible ; jamais une archive SHA256 validée. \`HEAD_404_NOT_CONFIRMED\` et \`UNVERIFIED\` = à examiner avant ingestion.

**Après V22 :** vérifier checksum, schémas et horodatages de chaque archive réellement ingérée, créer manifestes immuables pour 4 cryptos × 21 mois, et préserver les archives sources avant toute modélisation B01/B02. Historique exploratoire déjà inspecté, pas de nouveau OOS, aucun ordre.
