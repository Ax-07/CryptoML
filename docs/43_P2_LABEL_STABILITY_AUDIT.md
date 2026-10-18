# P2 — Audit descriptif H2/H6 des labels et de leur stabilité

## Objectif

Avant de rechercher un meilleur classificateur, mesurer si la cible `UP/NEUTRAL/DOWN` est naturellement dominée par le bruit : proportion des trois classes, amplitude des rendements, proximité de la barrière et sensibilité du label à des variations *descriptives* de la barrière. La référence reste **`vol_scaled_v1`, multiplicateur 0,5**, avec seuil instantané `0.5 × volatility_48(t) × sqrt(horizon)` et rendement `close(t+H)/close(t)-1`. Aucun changement de cible n'est décidé par ce rapport.

Le programme `mlpipeline.label_stability_audit` utilise le rapport existant `p2_feature_reduction_individual_remaining25_full_v1.json` pour lire les bornes et les effectifs des checks de développement, y compris la purge de maturité H12 (60 minutes). Les bougies des 4 actifs déjà vus sont lues à partir des archives P0 **en lecture seule**, puis tronquées à la dernière borne de check autorisée **avant** toute construction de feature ou de label ; seules les lignes des checks de développement sont résumées. Les tests OOS de chaque fold, les 90 jours finaux réservés et ADA ne sont ni évalués ni inclus dans les comptes. Le programme vérifie l'empreinte des archives et le nombre d'observations originales par check ; refuse l'écrasement du rapport ; n'entraîne rien et ne charge aucun modèle P3.

Attention : les checks ont déjà été consultés à plusieurs reprises ; les folds sont dépendants ; les seuils alternatifs ne constituent **pas** des candidats validés. Les comparaisons rétrospectives de labels ne mesurent **aucune** amélioration de prédiction. Le calcul de fréquence directionnelle majoritaire *parmi les vrais mouvements* est un oracle descriptif (l'événement mouvement n'est pas connu à la prédiction).

## Commandes PowerShell

Dans le checkout isolé :

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Smoke technique 1 fold uniquement (le résultat n'est pas une mesure scientifique représentative) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.label_stability_audit `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_label_stability_h2_h6_smoke_v1.json" `
  --months 24 --max-folds 1 --verify-hashes
```

Full : les 36 checks historiques purgés (si les tests et le smoke ont réussi) :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.label_stability_audit `
  --data-root "E:\cryptoML\data" `
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_label_stability_h2_h6_full_v1.json" `
  --months 24 --verify-hashes
```

## Lecture du bilan

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_label_stability_h2_h6_full_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = foreach ($h in @("H2","H6")) {
  $s = $r.summary.$h
  [PSCustomObject]@{
    Horizon = $h
    DOWN_Pct = [math]::Round(100*$s.mean_fold_class_fractions.DOWN,2)
    NEUTRAL_Pct = [math]::Round(100*$s.mean_fold_class_fractions.NEUTRAL,2)
    UP_Pct = [math]::Round(100*$s.mean_fold_class_fractions.UP,2)
    PresBarrierePct = [math]::Round(100*$s.mean_fold_near_barrier_fraction,2)
    ChangeA025Pct = [math]::Round(100*$s.threshold_sensitivity_mean_fold_changed_fraction.'0.25',2)
    ChangeA075Pct = [math]::Round(100*$s.threshold_sensitivity_mean_fold_changed_fraction.'0.75',2)
    ChangeA100Pct = [math]::Round(100*$s.threshold_sensitivity_mean_fold_changed_fraction.'1.0',2)
  }
}
$summary | Format-Table -AutoSize
```

Les résultats sont **des diagnostics de la cible**, pas des mesures de précision du modèle ou de rentabilité, et n'autorisent aucune promotion automatique, aucune relance P3 ou mesure sur la réserve finale.
