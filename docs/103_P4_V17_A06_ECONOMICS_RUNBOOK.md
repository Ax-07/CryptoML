# P4 V17 — Exécution Windows de l'audit économique A06

Protocole préenregistré : [102_P4_V17_A06_FROZEN_ECONOMIC_PROTOCOL.md](102_P4_V17_A06_FROZEN_ECONOMIC_PROTOCOL.md).

**V16 validée par l'opérateur :** cinq tests spécifiques réussis, suite complète à 100 %, quatre actifs P0 contrôlés et **8 833 signaux** A06 LONG/SHORT. Aucun rendement A06 n'a été consulté pour choisir le mécanisme V17. V17 ne modifie pas les règles de signal V16.

## Installer V17

```powershell
cd E:\cryptoML-volume-funding-v16
git fetch origin
git worktree add E:\cryptoML-volume-economics-v17 origin/research/a06-volume-economics-v17
cd E:\cryptoML-volume-economics-v17\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_volume_taker_economics_v17.py tests/test_volume_funding_readiness_v16.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**STOP si le moindre test échoue.** Les tests sont publiés mais n'ont pas encore été exécutés dans l'environnement Windows de l'opérateur.

## Lancer l'audit V17 seulement après tests valides

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.volume_taker_economics_v17 `
  --data-root "E:\cryptoML\data" `
  --reference-json "E:\cryptoML-walkforward-results\p4_v16_volume_readiness_v1.json" `
  --reference-csv "E:\cryptoML-walkforward-results\p4_v16_a06_signals_v1.csv" `
  --output-json "E:\cryptoML-walkforward-results\p4_v17_a06_economics_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v17_a06_events_v1.csv"
```

Les sorties **doivent être nouvelles** et hors P0. V17 revalide exactement les anciens signaux V16 contre les 84 Parquets, puis simule **12 M5 après la clôture du signal**, avec stop 1 ATR14 M15, target 1,5 ATR14 M15 et coûts hypothétiques 2/5/10 bp. Sorties `LONG / SHORT / ALL` et ventilation par actif. Stop si référence modifiée, trous M5, horizon insuffisant ou données traversant juillet 2026.

**Attention :** aucun frais funding, slippage, spread variable, position simultanée, allocation de capital ni vrai prix d'exécution. Une moyenne par signal n'est pas un PnL de portefeuille. La période étudiée était déjà explorée : aucune démonstration OOS indépendante.

## Funding et basis : audit séparé

Le mode V16 `--probe-funding-history` est facultatif et n'est pas requis pour l'économique A06. Il effectue des requêtes publiques limitées pour tester l'existence de quelques vieux taux de funding ; cela ne valide pas un arbitrage. Construire B01/B02 nécessitera des historiques complets spot/perp synchronisés avec risque et frictions des deux jambes.

**Fournir les sorties des tests puis la console V17. Ne pas activer V10 live, ne pas entraîner de sélecteur multi-stratégies pour l'instant.**
