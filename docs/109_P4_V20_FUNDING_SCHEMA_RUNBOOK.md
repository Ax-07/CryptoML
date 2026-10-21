# CryptoML V20 — Audit du schéma funding BTCUSDT, juin 2025

V19 est validé par l'opérateur : 6 tests dédiés + suite complète sans échec ; spot/perp/mark = 8640 M5 chacun, contrôlés avec SHA256 et schéma. Funding était `SCHEMA_REVIEW_REQUIRED`. Le V19 JSON contient le nombre de timestamps M5 réellement alignés, non affiché dans la console.

## Installer V20

```powershell
cd E:\cryptoML-funding-integrity-v19
git fetch origin
git worktree add E:\cryptoML-funding-schema-v20 origin/research/funding-archive-schema-v20
cd E:\cryptoML-funding-schema-v20\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_archive_schema_v20.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Ne pas lancer la sonde si les tests échouent.** Les tests V20 sont publiés mais restent à confirmer sur la machine Windows de l'opérateur.

## Audit du vrai schéma du ZIP de funding

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_archive_schema_v20 `
  --inspect-funding-archive `
  --reference-v19-json "E:\cryptoML-walkforward-results\p4_v19_archive_integrity_btc_2025_06_v1.json" `
  --output-json "E:\cryptoML-walkforward-results\p4_v20_funding_schema_btc_2025_06_v1.json"
```

La commande lit les résultats V19 et télécharge **uniquement l'archive funding BTC juin 2025 et son texte CHECKSUM**, avec limites d'octets. Elle vérifie SHA256, les membres ZIP, le schéma CSV, les timestamps de paiement et le champ de taux en fraction décimale. Le module gère deux schémas explicites et reporte `SCHEMA_REVIEW_REQUIRED` si les champs ne sont pas reconnus. Aucun OHLC ni taux individuel n'est conservé dans les sorties. Le JSON est nouveau et hors P0.

Même en cas de succès, **aucun historique des 21 mois × 4 symboles n'est encore certifié** ; aucune expérience de carry/basis n'est autorisée sur la base du seul pilote. La prochaine étape sera une ingestion immuable de toutes les sources, avec inventaire complet et validation temporelle avant toute évaluation économique. Aucun ordre ni entraînement ML.
