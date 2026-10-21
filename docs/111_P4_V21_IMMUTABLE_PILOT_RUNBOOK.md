# V21 — Ingestion pilote immuable spot / perp / mark / funding

V20 a été **validée par l'opérateur** : dix tests dédiés, suite complète sans échec et archive funding BTCUSDT juin 2025 conforme (\`calc_time\`, \`funding_interval_hours\`, \`last_funding_rate\`, 90 événements). V19 avait déjà validé 8640 bougies M5 spot, perp et mark. L'évaluation économique des stratégies B01/B02 reste **interdite** à ce stade.

Lire d'abord : [110_P4_V21_BASIS_IMMUTABLE_PILOT_PROTOCOL.md](110_P4_V21_BASIS_IMMUTABLE_PILOT_PROTOCOL.md).

## Installation Windows

\`\`\`powershell
cd E:\cryptoML-funding-schema-v20
git fetch origin
git worktree add E:\cryptoML-funding-pilot-v21 origin/research/funding-basis-archive-integrity-v21
cd E:\cryptoML-funding-pilot-v21\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_immutable_pilot_v21.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Ne pas lancer l'ingestion si un seul test échoue.** Ces tests sont nouveaux et n'ont pas encore été validés sur la machine Windows.

## Pilotage volontaire (lecture Binance, stockage hors P0)

Cette commande ne cible que **BTCUSDT juin 2025**, quatre archives et quatre CHECKSUM, mais peut télécharger plusieurs dizaines de Mo. Elle sauvegarde les quatre archives brutes et produit deux tables Parquet (spot/perp/mark joints M5, événements funding séparés) avec une chaîne SHA256 et un manifeste. Le répertoire de destination doit être **inexistant** et situé hors de \`E:\cryptoML\data\`.

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_immutable_pilot_v21 \`
  --ingest-verified-pilot \`
  --reference-v19-json "E:\cryptoML-walkforward-results\p4_v19_archive_integrity_btc_2025_06_v1.json" \`
  --reference-v20-json "E:\cryptoML-walkforward-results\p4_v20_funding_schema_btc_2025_06_v1.json" \`
  --data-root "E:\cryptoML\data" \`
  --destination "E:\cryptoML-walkforward-results\p4_v21_btc_2025_06_immutable"
\`\`\`

### Critères de validation

- SHA256 de chaque ZIP et de son CHECKSUM officiel vérifiés ;
- quatre schémas correctement reconnus (sinon blocage, sans remplir artificiellement le manque) ;
- 8640 M5 par marché, alignées à l'heure UTC et sans gaps ; OHLC perp en accord avec P0 immuable ;
- 90 événements de funding réels stockés **à part**, sans ffill ni utilisation prédictive ;
- manifest SHA256 des sources, Parquets et rapports V19/V20 ; un dossier final uniquement si tout a réussi.

**Une anomalie de colonnes Binance ou de \`close_time\` devra être examinée à partir des données réellement obtenues, jamais ignorée pour faire passer l'ingestion.** Les tests synthétiques ne prouvent pas à eux seuls la compatibilité du vrai CSV.

Cette sortie est un pilote d'intégrité, **pas un backtest, ni un gain potentiel démontré**. Les travaux sur le carry delta-neutre exigent ensuite l'inventaire complet des archives multi-mois, la synchronisation causale, le modèle de deux jambes et des frais/risques exécutables.
