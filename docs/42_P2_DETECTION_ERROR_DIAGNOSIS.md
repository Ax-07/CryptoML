# P2 — Diagnostic des erreurs de détection avant nouveaux modèles

Les gains d'ablation H2/H6 sont faibles. Avant tout test en conditions réelles ou toute nouvelle sélection de features, commencer par le **diagnostic descriptif** des erreurs de la référence immuable `baseline_30`.

## Données et périmètre

Le programme `mlpipeline.detection_error_audit` lit **uniquement** le JSON P2 existant `p2_feature_reduction_individual_remaining25_full_v1.json`. Il ne lit pas les Parquet, ne charge aucun modèle, ne réentraîne rien et ne contacte aucun flux en temps réel. Les 36 checks historiques ont été maintes fois inspectés ; ils ne sont pas indépendants. Il n'évalue ni ADA, ni tests OOS réservés, ni 90 derniers jours réservés.

Pour H2 et H6, le programme sépare les décisions directionnelles sélectionnées par le seuil déjà fixé en trois catégories : correctement directionnelles, réellement NEUTRAL (mouvement trop faible au sens du label) et réellement direction opposée. Il détaille les résultats par BTC/ETH/SOL/XRP, par UP/DOWN, couverture, précision, référence aléatoire pondérée, dispersion entre folds, cinq folds les plus faibles et différence descriptive entre les moitiés chronologiques. Il n'invente pas de nouvelles prédictions ; la décomposition est calculable à partir des comptes déjà conservés.

**Convention importante :** `uplift_selected_weighted_points` est pondéré par le nombre de décisions sélectionnées, à distinguer de la moyenne d'uplift des folds ; les observations et périodes sont réutilisées. Les résultats servent à poser des hypothèses, pas à démontrer qu'un changement améliorerait les performances futures.

## Exécution PowerShell

Depuis `E:\cryptoML-walkforward-test\backend` après synchronisation de la branche expérimentale :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.detection_error_audit `
  --source "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" `
  --output "E:\cryptoML-walkforward-results\p2_h2_h6_detection_error_audit_v1.json"
```

Les rapports existants ne sont jamais écrasés. Ne pas relancer si le rapport de sortie existe déjà.

Lecture rapide :

```powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_h2_h6_detection_error_audit_v1.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$lines = foreach ($h in @("H2","H6")) {
    $p = $r.horizons.$h.overall
    [PSCustomObject]@{
        Horizon = $h
        PrecisionPct = [math]::Round(100*$p.precision_selected,2)
        ReferencePct = [math]::Round(100*$p.random_reference_selected_weighted,2)
        UpliftPoints = [math]::Round($p.uplift_selected_weighted_points,3)
        CoveragePct = [math]::Round(100*$p.coverage_realized,2)
        NeutralParmiDecisionsPct = [math]::Round(100*$p.fraction_of_selected_neutral_errors,2)
        OpposeParmiDecisionsPct = [math]::Round(100*$p.fraction_of_selected_opposite_errors,2)
        FoldPositifs = $p.folds_above_reference
    }
}
$lines | Format-Table -AutoSize
```

**Prochaine décision seulement après ce diagnostic :** si le coût vient surtout de faux mouvements `NEUTRAL`, examiner la définition des labels, la sensibilité au bruit et la sélection de mouvements significatifs ; s'il vient surtout des directions inversées, étudier des représentations de régime ou un modèle séparant détection de mouvement et choix UP/DOWN. Ces idées nécessiteraient un protocole d'étude historique propre, sans utiliser les périodes scellées, et aucune promotion automatique. La disponibilité des informations intrabar et des coûts d'exécution doit être étudiée séparément. P3 et les deux modèles H6 gelés restent inchangés.
