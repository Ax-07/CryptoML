# P4 — Recensement des pullbacks EMA9/EMA26 (sans labels)

**Objectif :** compter les premiers contacts EMA9 détectés par `trend_pullback.py`, sans aucune mesure de profit, sans fit ML et sans ouverture des périodes de test. Rapport exploratoire uniquement.

Périmètre : **BTC, ETH, SOL, XRP**, partitions mensuelles P0 de **décembre 2025 à juin 2026 inclus**, toutes vérifiées par empreinte SHA-256. Le chargeur réutilisé `prospective_model_pack.load_clean_historic` n'ouvre aucun Parquet de juillet 2026 ou ultérieur ; ni ADA, ni P2, ni P3, ni OOS réservé.

Le rapport indique, pour chaque actif et mois : bougies clôturées, indicateurs prêts, tendance haussière t-1, premier contact EMA9, candidats bruts, candidats conservés après cooldown de 8 bougies, taux pour 1000 bougies. Les sommes mensuelles sont réconciliées avec les totaux. Aucun `pullback_outcomes` n'est importé.

## Vérifier sur Windows

```powershell
cd E:\cryptoML-trend-pullback
git fetch origin
git status --short
# En cas de modifications locales non validées, les préserver avant toute mise à jour.
git switch -c research/trend-pullback-ml-v1 --track origin/research/trend-pullback-ml-v1
# Si cette branche existe déjà localement, utilisez plutôt git switch research/trend-pullback-ml-v1
git pull --ff-only origin research/trend-pullback-ml-v1
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_frequency.py tests/test_trend_pullback.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Si le worktree est en HEAD détachée et que la branche existe déjà sur un autre worktree local, ne pas forcer la bascule : faire `git fetch origin` puis `git switch --detach origin/research/trend-pullback-ml-v1` dans ce worktree propre.

## Mesurer les fréquences (lecture seule de P0)

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_frequency `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_pullback_frequency_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_pullback_frequency_v1.csv"
```

Les deux sorties doivent être **nouvelles** et hors de `E:\cryptoML`. Le script refuse un écrasement.

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p4_pullback_frequency_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.by_asset.PSObject.Properties | ForEach-Object {
    [PSCustomObject]@{
        Crypto = $_.Name
        Bars = $_.Value.all.closed_bars
        Brut = $_.Value.all.raw_candidates
        Retenus = $_.Value.all.cooldown_selected
        Pour1000 = $_.Value.all.selected_per_1000_bars
    }
} | Format-Table -AutoSize
Import-Csv "E:\cryptoML-walkforward-results\p4_pullback_frequency_v1.csv" |
    Format-Table -AutoSize
```

**Ne pas optimiser le détecteur en fonction des performances rétrospectives** lors de ce recensement. Si trop peu d'événements existent, documenter le constat et concevoir un nouveau protocole V2 avant de regarder des résultats de trading. Les checks, holdouts précédemment consultés et données futures ne constituent pas une confirmation indépendante.
