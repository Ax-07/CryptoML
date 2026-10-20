# P2 — Réduction exploratoire des 30 features historiques

Les expériences de structure HH/HL/LH/LL n'ont pas établi de gain robuste. Cette étude revient aux **30 variables actuelles**, sans ajouter de colonnes ni modifier les modèles P3 déjà figés.

## Protocole

- `--mode group` : 30 features vs suppression de chacune des 7 familles existantes (8 variantes × 2 horizons × 36 folds = **576 fits**).
- `--mode individual` : 30 features vs suppression d'une feature à la fois ; le programme accepte `--only-features ret_1 ret_3` pour limiter explicitement le nombre d'essais. Sans restriction, 31 variantes × 2 × 36 = **2 232 fits**.
- Mêmes 4 cryptos BTC/ETH/SOL/XRP, 120 jours train, 7 jours early stop, 3 jours tuning antérieur, 4 jours validation descriptive ; 14 jours test OOS **non consultés**, derniers 90 jours **non évalués**, ADA **jamais chargé**.
- Purge H12 de la maturité des labels ; seuil directionnel visant **40 % des observations sur le tuning non étiqueté**. La couverture du check n'est pas contrainte artificiellement ; folds à seuil irréalisable marqués comme tels.
- Dans `paired_drop_vs_full`, un `mean_uplift_delta_points` **positif** signifie qu'enlever des features a amélioré l'uplift moyen par rapport au baseline, sur les folds où les deux variantes étaient évaluables. Lire aussi `paired_feasible_folds`, `positive_uplift_folds`, `mean_realized_coverage_delta_points`, `mean_log_loss_improvement` et les détails par actif.
- Les checks historiques ont été inspectés à plusieurs reprises ; les résultats sont **exploratoires, pas une validation indépendante**. Aucune suppression automatique ni décision de promotion.

## Commandes PowerShell

Depuis `E:\cryptoML-walkforward-test\backend`, après `git fetch origin` et `git switch --detach origin/feat/cpu-purged-walkforward-folds` :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Smoke familles** (1 fold ; 16 fits) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.feature_reduction_diagnostics `
  --mode group --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_feature_reduction_group_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

**Étude complète familles**, uniquement après smoke vert (576 fits) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.feature_reduction_diagnostics `
  --mode group --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_feature_reduction_group_full_v1.json" `
  --months 24 --n-estimators 150 --seed 42 --verify-hashes
```

**Mode individuel**, plus tard, sur 2 variables précises (6 fits smoke) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.feature_reduction_diagnostics `
  --mode individual --only-features ret_1 ret_3 `
  --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

Les fichiers existants ne sont jamais écrasés. Résumés PowerShell :

```powershell
$p = Get-Content "E:\cryptoML-walkforward-results\p2_feature_reduction_group_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$p.summary.H2.paired_drop_vs_full | ConvertTo-Json -Depth 10
$p.summary.H6.paired_drop_vs_full | ConvertTo-Json -Depth 10
```
