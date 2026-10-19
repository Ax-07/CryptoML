# P2 — Audit des désaccords LightGBM vs réversion (H2/H6)

## Question

Sur les **checks de développement déjà inspectés**, quelle fraction de la différence entre LightGBM et les règles simples provient de corrections utiles ou d'erreurs supplémentaires ? Deux comparaisons choisies **après inspection** des résultats précédents : H2 contre réversion 3 bougies (M15) ; H6 contre réversion 6 bougies (M30). **Ce choix adaptatif interdit de traiter le résultat comme une confirmation indépendante.**

## Six situations mutuellement exclusives

| Situation | Interprétation |
|---|---|
| Accord correct | Même sens prédit, bonne direction |
| Accord erroné NEUTRAL | Même sens, mais cible neutre |
| Accord erroné opposé | Même sens, direction réelle inverse |
| Désaccord : ML correct | Correction utile par LightGBM |
| Désaccord : réversion correcte | Erreur ajoutée par LightGBM |
| Désaccord : deux erreurs NEUTRAL | Deux sens opposés, mais aucun mouvement étiqueté |

Ces six classes partitionnent **toutes les observations sélectionnées**. Pour toute vue et tout sous-groupe, le gain net du ML se déduit exclusivement du nombre de désaccords avec ML correct moins le nombre de désaccords avec réversion correcte. Aucun événement n'est enregistré en détail : seulement des compteurs par fold, horizon, crypto et régime LOW/MID/HIGH. Les quantiles de volatilité par actif sont calculés **sur le train uniquement**.

**Deux vues :** `all_check` (une direction à chaque bougie, sans abstention) et `shared_ml_selected` (couverture cible 40 % sur les scores de tuning du ML uniquement, couverture effective variable en check). La réversion sur le masque ML n'est **pas** une stratégie autonome à 40 %.

## Sécurité et reproductibilité

- Le protocole original fournit les limites des 36 folds, les effectifs et l'empreinte du dataset ; le JSON du benchmark directionnel existant sert de **référence immuable**. Pour chaque fold, le code tronque les features **avant de générer les labels** à la fin du check de ce fold : jamais de labels de son test OOS.
- Les 4 actifs sont BTCUSDT, ETHUSDT, SOLUSDT et XRPUSDT. Aucun ADA, ni labels du dernier holdout de 90 jours, ni modification de P3 ou des modèles H6 figés.
- 120 jours de train, 7 jours d'early stop, 3 jours de tuning pour un seuil fondé sur les scores **sans consulter les labels tuning**, 4 jours de check, purge de maturité H12 (60 minutes). H2 et H6 gardent la même cible et les mêmes 30 features LightGBM, 150 estimateurs CPU, graine 42.
- Le nouveau rapport **doit reproduire pour chaque fold** les effectifs et les comptes corrects du ML et de la réversion présents dans le benchmark d'origine. Si ce n'est pas le cas, arrêter et diagnostiquer ; ne pas accepter silencieusement des résultats différents.
- Les mêmes périodes de développement ont été consultées plusieurs fois. Les folds se recoupent dans le train et les marchés sont corrélés. Il ne s'agit ni d'une validation nouvelle, ni d'une mesure de PnL ou de profit.

## Exécution dans le checkout isolé Windows

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

### Smoke (1 fold / 2 fits CPU)

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.directional_disagreement_audit `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_directional_disagreement_smoke_v1.json" `
  --months 24 --max-folds 1 --verify-hashes
```

### Full (36 folds, 72 fits CPU) seulement si tests et smoke passent

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.directional_disagreement_audit `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --baseline-report "E:\cryptoML-walkforward-results\p2_directional_rule_benchmark_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_directional_disagreement_full_v1.json" `
  --months 24 --verify-hashes
```

## Lecture rapide

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_directional_disagreement_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = foreach ($h in @("H2","H6")) {
  $m = $r.summary.$h.shared_ml_selected
  [PSCustomObject]@{
    Horizon=$h
    Folds=$m.fold_records_with_rows
    Observations=$m.rows
    AccordPct=[math]::Round(100*$m.agreement_rate,2)
    DesaccordPct=[math]::Round(100*$m.disagreement_rate,2)
    CorrectionML=$m.counts.disagree_ml_correct
    CorrectionReversion=$m.counts.disagree_reversion_correct
    DesaccordNeutre=$m.counts.disagree_both_wrong_neutral
    GainMLPts=[math]::Round($m.net_gain_ml_points,3)
  }
}
$rows | Format-Table -AutoSize
```

Le gain pondéré par toutes les décisions (`net_gain_ml_points`) peut être légèrement différent de la **moyenne non pondérée des gains par fold** (`mean_fold_gain_ml_points`) déjà citée. Les deux métriques sont enregistrées et ne doivent pas être confondues.

**Étape suivante après les comptes :** décrire la fréquence et la qualité des désaccords, regarder leur stabilité par actif/régime ; ne **pas** choisir un filtre ou entraîner une nouvelle architecture sur les checks exploratoires sans plan de confirmation indépendante.