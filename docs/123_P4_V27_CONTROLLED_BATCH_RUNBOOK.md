# V27 — Lots contrôlés 1 à 4 partitions, étape de planification

V26 a été confirmé sur l'ordinateur de l'opérateur : XRPUSDT octobre 2024 et ETHUSDT novembre 2024 vérifiés, total **8/84**, **76 manquantes**. V27 a été publié mais ses nouveaux tests ne sont **pas encore exécutés sur Windows**.

[Protocole de contrôle V27](122_P4_V27_CONTROLLED_BATCH_PROTOCOL.md).

## Installer et exécuter les tests d'abord

```powershell
cd E:\cryptoML-flexible-batches-v26
git fetch origin
git worktree add E:\cryptoML-controlled-batches-v27 origin/research/funding-basis-controlled-batches-v27
cd E:\cryptoML-controlled-batches-v27\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_controlled_batches_v27.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Si un seul test échoue, STOP.** Partager la traceback sans autoriser l'ingestion.

## Phase A : produire le plan, strictement sans réseau

Choix préenregistré de 4 partitions manquantes : BTCUSDT:2024-11, SOLUSDT:2024-11, XRPUSDT:2024-11, ETHUSDT:2024-12.

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_controlled_batches_v27 `
  --mode plan `
  --target BTCUSDT:2024-11 --target SOLUSDT:2024-11 `
  --target XRPUSDT:2024-11 --target ETHUSDT:2024-12 `
  --data-root "E:\cryptoML\data" `
  --v21-directory "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable" `
  --v23-root "E:\cryptoML-walkforward-results\p4_v23_verified_sources" `
  --coverage-json "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.json" `
  --coverage-csv "E:\cryptoML-walkforward-results\p4_v22_archive_coverage_full_v1.csv" `
  --output-plan "E:\cryptoML-walkforward-results\p4_v27_batch_plan_v1.json"

$planSha = (Get-FileHash "E:\cryptoML-walkforward-results\p4_v27_batch_plan_v1.json" -Algorithm SHA256).Hash
Write-Host "V27 SHA256 =" $planSha
```

Plan attendu : 8 verified, 76 missing, quatre cibles exactes. Il faut **examiner la sortie avant d'autoriser les GET**.

## Phase B : seulement après revue et validation du plan

Le SHA exact calculé par PowerShell est obligatoire, sans valeur inventée. Jusqu'à 16 ZIP et 16 CHECKSUM publics, téléchargés un à un par partition ; chaque partition est soumise aux mêmes contrôles V23 que BTC/ETH/SOL/XRP précédemment.

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_controlled_batches_v27 `
  --mode execute --permit-downloads `
  --plan-json "E:\cryptoML-walkforward-results\p4_v27_batch_plan_v1.json" `
  --plan-sha256 $planSha `
  --journal-jsonl "E:\cryptoML-walkforward-results\p4_v27_batch_journal_v1.jsonl" `
  --summary-json "E:\cryptoML-walkforward-results\p4_v27_batch_summary_v1.json"
```

**Stop si le disque libre est inférieur à 512 MiB** avant chaque nouvelle partition. Fichiers de plan, journal et résumé toujours nouveaux ; aucune suppression ou écrasement des sources validées. Si interruption, réexécuter le **même plan** avec nouveaux chemins journal et résumé : toutes les partitions déjà conformes seront sautées après audit. Premier échec => arrêt et journalisation.

En cas de succès réel de quatre partitions, total attendu **12/84 verified**, **72 missing**, jamais supposé par avance. Ce ne sont pas des rendements de trading. Les données sont celles du développement historique déjà inspecté, pas des observations indépendantes.
