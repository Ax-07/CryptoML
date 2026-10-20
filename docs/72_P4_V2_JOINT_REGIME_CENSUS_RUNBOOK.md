# P4 V2 — Comptage croisé force × profondeur (sans labels)

Complément strictement **descriptif** à `docs/70_P4_V2_TREND_STRENGTH_DEPTH_PROTOCOL.md` et au recensement de `docs/71_P4_V2_REGIME_CENSUS_RUNBOOK.md`.

On compte les 12 combinaisons des 3 classes de force EMA9−EMA26 / ATR et 4 classes de profondeur EMA9−low / ATR déjà fixées. Une classe `no_undercut` à zéro est attendue par construction du détecteur V1. Résultats par mois/crypto, avec réconciliation automatique aux totaux V2, et vérification SHA-256 des partitions P0. Aucun label, rendement, PnL, apprentissage ou choix de classe favorable. Quatre cryptos vues uniquement ; décembre 2025 à juin 2026 seulement, hors ADA et fenêtres réservées.

## Commandes

```powershell
cd E:\cryptoML-trend-pullback-v2
git status --short
git fetch origin
# Exécuter uniquement si le worktree est propre :
git switch --detach origin/research/trend-pullback-regime-v2
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_regime_joint_census_v2.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_regime_joint_census_v2 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v2_joint_census_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v2_joint_census_v1.csv"
```

Les deux fichiers de sortie doivent être **nouveaux**.

```powershell
$r=Get-Content "E:\cryptoML-walkforward-results\p4_v2_joint_census_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.total | Format-List
Import-Csv "E:\cryptoML-walkforward-results\p4_v2_joint_census_v1.csv" |
  Where-Object { [int]$_.events -gt 0 } |
  Sort-Object symbol,month,strength,depth |
  Format-Table -AutoSize
```

**Suite :** analyser la suffisance des effectifs et la représentation des douze combinaisons, **pas leurs rendements**. Un éventuel protocole prédictif ultérieur doit être défini dans un document distinct avant d'ouvrir les labels V2. Les données historiques déjà étudiées ne sont pas indépendantes.
