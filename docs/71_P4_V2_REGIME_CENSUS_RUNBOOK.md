# P4 V2 — Recensement descriptif sans labels

Le module `mlpipeline.pullback_regime_census_v2` applique le détecteur P4 V1 inchangé, calcule les dix features causales V2, puis compte exclusivement les **signaux** par classe de force et de profondeur (seuils figés dans `docs/70_P4_V2_TREND_STRENGTH_DEPTH_PROTOCOL.md`). Pas de rendements, pas de labels, pas d'entraînement, pas de choix de bucket « gagnant ».

Sources : exclusivement les partitions P0 **décembre 2025 à juin 2026**, quatre actifs BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT. Le chargeur réutilisé vérifie les sommes SHA-256, les bougies clôturées et n'ouvre pas ADA ou juillet–septembre. Les mois avec zéro signal sont représentés explicitement. Totaux mensuels, par actif, et globaux réconciliés. Les données restent historiques réutilisées, **jamais OOS indépendantes**.

## Windows / worktree V2

```powershell
cd E:\cryptoML-trend-pullback-v2
git status --short
git fetch origin
# Seulement si aucun changement local à préserver :
git switch --detach origin/research/trend-pullback-regime-v2
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_regime_census_v2.py tests/test_pullback_regime_v2.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_regime_census_v2 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v2_regime_census_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v2_regime_census_v1.csv"
```

Les noms de sortie doivent être **inexistants** et placés en dehors de P0. Aucune donnée de marché ni ancienne sortie n'est modifiée.

## Lire les agrégats

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p4_v2_regime_census_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.total | Format-List
$r.by_asset.PSObject.Properties | ForEach-Object {
    [PSCustomObject]@{
       Crypto = $_.Name
       Events = $_.Value.total.events
       Classified = $_.Value.total.valid_events
       Unclassified = $_.Value.total.unclassified
       LowTrend = $_.Value.total.strength.low
       MidTrend = $_.Value.total.strength.mid
       HighTrend = $_.Value.total.strength.high
       ShallowPullback = $_.Value.total.depth.shallow
       MediumPullback = $_.Value.total.depth.medium
       DeepPullback = $_.Value.total.depth.deep
    }
} | Format-Table -AutoSize
Import-Csv "E:\cryptoML-walkforward-results\p4_v2_regime_census_v1.csv" |
  Format-Table -AutoSize
```

La prochaine décision après recensement concerne uniquement la qualité de la distribution des observations, **pas la rentabilité des classes**. Aucun nouvel entraînement ou backtest tant qu'un protocole séparé n'est pas figé.
