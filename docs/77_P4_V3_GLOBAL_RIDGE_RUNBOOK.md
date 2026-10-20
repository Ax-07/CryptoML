# P4 V3 — Ridge global 18 mois, protocole figé (exécution)

Source méthodologique inchangée : `docs/76_P4_V3_21MONTH_GLOBAL_RIDGE_PROTOCOL.md`.
**Cette exécution est EXPLORATOIRE ; mai/juin 2026 ne sont pas un OOS indépendant** et les fills OPEN(t+2) restent théoriques. Aucun ajustement des résultats après lecture.

## Trois modèles et deux comparateurs

- `C0_ridge9_short4m` : 9 variables d'origine, train décembre 2025–mars 2026 ;
- `C1_ridge9_long18m` : les mêmes 9 variables, train octobre 2024–mars 2026 ;
- `C2_ridge19_long18m` : 9 variables de confirmation au close(t+1) + 10 caractéristiques de régime V2 au close(t), même train de 18 mois ;
- `A_all`, `B_confirmed`, `D_no_trade` (net=0) ;
- chacun des trois modèles comparé à l'**espérance analytique d'un tirage uniforme** de même nombre de transactions parmi les B **au sein de chaque crypto×mois**. Ce n'est pas un test statistique ou une p-value.

`StandardScaler + Ridge(alpha=10)`, cible théorique net à 5bp avec clipping à ±100bp uniquement sur train. Les trois modèles utilisent la même calibration d'avril (quantile 80 des scores et plancher 0), qui **n'utilise pas les issues d'avril** ; le seuil est strictement dépassé pour trader. Diagnostics mai/juin seulement, purge `mature_at < fin`. Les coûts 2/5/10bp sont reportés **sans adapter les choix**.

## Windows — exécuter les tests avant le premier calcul

```powershell
cd E:\cryptoML-trend-pullback-v3
git status --short
git fetch origin
# Si worktree propre :
git switch --detach origin/research/trend-pullback-long-history-v3

cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_global_ridge_v3.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Si tests verts, lancer **une fois**, sans grille :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_global_ridge_v3 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v3_global_ridge_18m_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v3_global_ridge_18m_v1.csv"
```

Les chemins de sortie doivent désigner deux **nouveaux fichiers** et ne sont jamais écrasés. Chargeur vérifié SHA-256 : seulement BTC/ETH/SOL/XRP de 2024-10 à 2026-06, interdit ADA, juillet–septembre et P3.

## Lire les résultats sans sélectionner a posteriori

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p4_v3_global_ridge_18m_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.models | Format-List
$r.combined_may_june | Format-List

Import-Csv "E:\cryptoML-walkforward-results\p4_v3_global_ridge_18m_v1.csv" |
  Select-Object month,symbol,strategy,opportunities,trades,net_5bps_per_trade,net_5bps_per_opportunity,random_expected_net_5bps_per_trade,edge_vs_random_5bps_per_trade |
  Format-Table -AutoSize
```

### Interprétation

La rentabilité nette totale **des stratégies** n'est pas prouvée par un gain moyen sur la seule sélection ; sans edge supérieur à zéro après hypothèses raisonnables de frais/fill/slippage, la référence `D_no_trade` domine. Le résultat des périodes et familles inspectées reste de la recherche interne et **ne peut pas autoriser du trading réel**. Toute confirmation nécessite de nouvelles prédictions futures traçables après gel.
