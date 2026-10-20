# P4 — Baselines A/B fixées, protocole exploratoire

**Hypothèse à examiner, pas un résultat :** filtrer les premiers pullbacks EMA9 par une confirmation haussière peut améliorer le markout théorique net **par opportunité** par rapport à les prendre tous.

**Univers :** P0 BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, M5 clôturées, partitions décembre 2025 à juin 2026 inclus vérifiées SHA-256. Réutilisation de données de développement historiques déjà explorées : **ceci n'est pas un OOS indépendant**. Aucun entraînement, aucune utilisation ADA, juillet 2026 ou ultérieur, aucune consultation P2/P3.

A = tous les événements mûrs détectés par le code EMA9/26 V1. Après décision close(t), entrée hypothétique OPEN(t+5 minutes), soit l'OPEN de la bougie t+2 (une bougie complète d'attente). Stop -1 ATR, cible +1.5 ATR, six barres suivies, priorité défavorable en cas de double contact, scénarios coûts 2, 5, 10 bp aller-retour.

B = seulement les mêmes opportunités A pour lesquelles, **à la clôture de la bougie t+1**, les conditions suivantes sont vraies : close(t+1) > open(t+1), close(t+1) > EMA9(t+1), close(t+1) > close(t). Même entrée hypothétique OPEN(t+2) que A et mêmes règles de sortie. Les événements rejetés par B comptent 0 dans le résultat **par opportunité** ; le résultat par trade est présenté séparément. B a une bougie de confirmation disponible après le signal A, mais avant l'entrée des deux références.

**Prudence :** entrée à l'OPEN futur non garantie, sans spread/slippage réalistes, aucun PnL de portefeuille, aucune promesse de profit. Les markouts peuvent se chevaucher entre actifs. Aucun seuil n'est adapté après examen des résultats ; la comparaison est descriptive. Ne pas recommencer sur un ensemble de configurations jusqu'à obtenir un résultat favorable.

## Exécuter sur Windows

Dans le worktree créé précédemment :

```powershell
cd E:\cryptoML-trend-pullback
git status --short
git fetch origin
git switch --detach origin/research/trend-pullback-ml-v1
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_baselines.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_baselines `
 --data-root "E:\cryptoML\data" `
 --output-json "E:\cryptoML-walkforward-results\p4_pullback_baselines_ab_v1.json" `
 --output-csv "E:\cryptoML-walkforward-results\p4_pullback_baselines_ab_v1.csv"
```

Ces noms de fichiers doivent être **neufs** ; le programme refuse l'écrasement. Si modifications locales présentes dans `git status`, ne pas faire `git switch` avant de les préserver.

### Lire le résultat

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p4_pullback_baselines_ab_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.overall | Format-List
$r.by_asset.PSObject.Properties | ForEach-Object {
    $a = $_.Value.overall.A_take_every_matured_pullback
    $b = $_.Value.overall.B_confirm_next_close_before_common_entry
    [PSCustomObject]@{
        Crypto = $_.Name
        A_trades = $a.trades
        B_trades = $b.trades
        B_coverage = $b.coverage
        A_net2_per_opportunity = $a.net_2bps_per_opportunity
        B_net2_per_opportunity = $b.net_2bps_per_opportunity
        A_net5_per_opportunity = $a.net_5bps_per_opportunity
        B_net5_per_opportunity = $b.net_5bps_per_opportunity
    }
} | Format-Table -AutoSize
```

Évaluer les résultats par actif et par mois, pas seulement la moyenne globale. Un résultat positif ici restera exploratoire, non indépendant.
