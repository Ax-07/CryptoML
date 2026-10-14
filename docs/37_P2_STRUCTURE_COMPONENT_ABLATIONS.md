# P2 Structure de marché — étude de fréquence et ablation interne

La précédente comparaison 30 vs 42 features n'a montré aucun gain robuste. Ce nouvel outil pose deux questions distinctes : **les impulsions BOS/CHoCH sont-elles suffisamment fréquentes ?** et **la famille d'informations est-elle utile séparément ?**

## Variantes pré-définies

- `baseline_30` — 30 features originelles
- `plus_state_34` — + HH, HL, LH, LL (états persistants, non-comptages de nouveaux pivots)
- `plus_bos_32` — + cassures BOS up/down
- `plus_choch_32` — + changements CHoCH up/down
- `plus_distance_age_34` — + distances et âges des deux derniers swings confirmés
- `plus_all_structure_42` — + les douze, comme référence de reproductibilité du rapport précédent

Chaque variante utilise les **mêmes observations**, la même purge de maturité H12, le même fold H2/H6 et les mêmes périodes train/early/tuning/check. Le seuil visant 40 % de sélection est ajusté **sans étiquettes** sur les scores du tuning antérieur ; couverture réalisée sur check toujours affichée. Les folds où 40 % est irréalisable ne sont pas masqués.

Les probabilités et les uplifts sont purement descriptifs sur les historiques P2 déjà inspectés. Aucun trading, aucun réglage des modèles figés P3, aucun accès à ADA ni aux 90 derniers jours réservés.

## Exécution sous Windows

Depuis `E:\cryptoML-walkforward-test\backend` après `git fetch origin` et `git switch --detach origin/feat/cpu-purged-walkforward-folds` :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q tests/test_market_structure_component_diagnostics.py
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Étape 1 — fréquence (aucun réentraînement)**

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.market_structure_component_diagnostics `
  --mode frequency --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_structure_components_frequency_v1.json" `
  --months 24 --verify-hashes
```

Inspecter `frequency.all_check_rows`, `frequency.primary_12_17_UTC` et `frequency.by_symbol`. Une valeur faible de `bos_any` ou `choch_any` peut expliquer leur intérêt limité, sans prouver qu'ils sont inutiles. Les barres des états HH/HL/LH/LL ne sont pas des événements de swing.

**Étape 2 — smoke limité (12 entraînements, non concluant)**

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.market_structure_component_diagnostics `
  --mode ablation --data-root "E:\cryptoML\data" `
  --output "E:\cryptoML-walkforward-results\p2_structure_components_smoke_v1.json" `
  --months 24 --max-folds 1 --n-estimators 150 --seed 42 --verify-hashes
```

**Étape 3 — tous les folds** : lancer la même commande sans `--max-folds`, avec `--output "E:\cryptoML-walkforward-results\p2_structure_components_ablation_v1.json"`. Il peut y avoir **432 entraînements CPU** (36 folds × H2/H6 × 6 variantes). Aucun écrasement de rapport n'est autorisé.

## Limites

Ces tests s'ajoutent à de multiples analyses des mêmes périodes de développement : les différences peuvent être dues au hasard. Ne pas promouvoir une feature dans les modèles gelés sur cette base. Une preuve indépendante nécessite plus tard de nouvelles données, pas de nouvelle lecture des mêmes validations.
