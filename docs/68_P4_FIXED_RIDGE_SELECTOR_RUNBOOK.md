# P4 — Ridge V1 expérimental, pullbacks confirmés

Statut : **recherche historique sur données déjà explorées**. Ne prétend ni valider un avantage financier, ni créer un nouveau holdout indépendant. Aucune opération réelle.

## Hypothèse et protocole définis avant ce run

Un filtre Ridge simple peut-il sélectionner parmi les événements de baseline B (confirmation EMA9 au close de t+1) un sous-ensemble plus performant, **net après 5 bp** de coûts théoriques, que B et **que l'abstention totale (=0)** ?

- Uniquement BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT ; partitions Binance P0 décembre 2025 à juin 2026, contrôle SHA-256 ; ni ADA ni juillet 2026 ou après.
- Détecteur de pullbacks et barrières de baseline B **inchangés**. Entrée théorique OPEN(t+2), après décision connue au close(t+1).
- 9 features causales figées dans `pullback_ml.FEATURES` (EMA spread et pentes normalisées par ATR, distance close/EMA9, ATR relatif, corps/amplitude/mèche de bougie de confirmation, profondeur pullback).
- Ridge(alpha=10) + StandardScaler, fit uniquement sur les B de **décembre 2025 à mars 2026**, étiquettes connues à maturité au sein de chaque partition, cible net_5bps clipsée à ±100 bp durant fit ; aucune recherche d'hyperparamètre.
- **Avril 2026** sert uniquement à établir le seuil fixe `max(0, quantile 80% des prédictions d'avril)`, avec observations mûres avant le 1er mai. Le seuil ne dépend pas des rendements d'avril.
- **Mai et juin 2026** sont deux diagnostics chronologiques exploratoires : ils ont déjà été consultés sous les stratégies A/B, donc **pas un test indépendant**. Ne pas régler les features, le seuil ou l'algorithme en fonction de ces mois.
- Évaluer les mêmes opportunités A, B, C = B filtré par Ridge, et D = aucune entrée. Résultats aux coûts aller-retour 2, 5, 10 bp, par crypto et mois. Une amélioration par rapport à B **mais restant négative** ne démontre aucune valeur économique.

### Commandes Windows

```powershell
cd E:\cryptoML-trend-pullback
git status --short
git fetch origin
# Seulement si le worktree est propre :
git switch --detach origin/research/trend-pullback-ml-v1
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_ml.py tests/test_pullback_baselines.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_ml `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_ridge_selector_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_ridge_selector_v1.csv"
```

Le script exige des noms de résultats non encore existants et ne modifie aucune donnée P0.

À lire :

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p4_ridge_selector_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r | Select-Object protocol, score_threshold_predicted_net_5bps, train_B_events, april_B_calibration_events
Import-Csv "E:\cryptoML-walkforward-results\p4_ridge_selector_v1.csv" |
  Select-Object month, symbol, strategy, trades, net_5bps_per_opportunity, net_5bps_per_trade |
  Format-Table -AutoSize
```

Si C n'a aucun trade, c'est un résultat acceptable (abstention). Si C reste négatif ou instable, **ne pas relancer une grille de paramètres** sur les mêmes mois. Périodes de validation réellement futures uniquement, modèles et décision gelés avant les signaux.
