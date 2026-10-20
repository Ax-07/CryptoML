# P2 — H2 sélectif, cible retardée : exécution hors ligne

Protocole : `docs/59_P2_H2_DELAYED_RETURN_ABSTAIN_PROTOCOL.md`. Recherche exploratoire, **36 checks P2 déjà inspectés**. Aucun test 72 heures, aucune commande réelle. P0 original en lecture seule, BTC ETH SOL XRP uniquement ; pas d'ADA, d'OOS réservé, de final90 ni de P3.

Deux régressions fixes : `regression_30` (LightGBM trente features) et `ridge_ema2` (régression Ridge sur deux EMA). **Cible distincte de H2 classique :** prix open de la bougie commençant cinq minutes *après* la clôture du signal, puis close de la même bougie, cinq minutes plus tard. La décision utilise seulement les informations de t. Entraînement sur train 120 jours, early 7 jours ; 3 jours tuning non utilisés ; 4 jours check scorés ; purge H12 maintenue.

Politique figée : LONG si prévision strictement supérieure à +3 bps, SHORT sous −3 bps, sinon **abstention**. Hypothèses de coût aller-retour 0/2/5 bps sur **positions théoriques réellement ouvertes** ; abstention = zéro. Sont rapportés le rendement net par opportunité et par trade, le taux de couverture et les résultats par crypto/fold. Le modèle H2 historique 30 features sans abstention est une référence à politique différente, scorée sur les mêmes rendements retardés.

**Limite :** les OHLCV ne permettent pas de garantir une entrée au prix open(t+5) ni une sortie au close(t+10). Ce calcul reste un markout indicatif, pas un backtest exécutable ni une preuve de rentabilité. Les folds se chevauchent historiquement.

## 1. Récupérer puis tester

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h2_delayed_selective.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

## 2. Lancer le diagnostic 36 folds, sans test 72 h

Si les tests passent :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h2_delayed_selective `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --h1h6-report "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_v1.json" `
  --predictions-parquet "E:\cryptoML-walkforward-results\p2_ema9_26_h1h6_full_predictions_v1.parquet" `
  --output-json "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_full_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_full_v1.csv" `
  --output-parquet "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_predictions_v1.parquet" `
  --verify-hashes
```

Les trois fichiers de sortie ne doivent **pas déjà exister**. Vérifier la disponibilité de toutes les dépendances PyPI du backend. Toute erreur de géométrie de fold, d'étiquette temporelle, de fingerprint ou de schéma doit faire échouer le rapport.

## 3. Afficher le résultat

```powershell
Import-Csv "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_full_v1.csv" |
  Where-Object { $_.cost_bps -eq '2' } |
  Select-Object model, signals, opportunities, coverage,
    gross_per_opportunity_bps, net_per_opportunity_bps, net_per_trade_bps |
  Format-Table -AutoSize

$r = Get-Content "E:\cryptoML-walkforward-results\p2_h2_delayed_selective_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.by_asset | Format-List
```

Ne pas choisir a posteriori la crypto, le seuil ou la configuration gagnante en qualifiant les mêmes checks de validation indépendante.
