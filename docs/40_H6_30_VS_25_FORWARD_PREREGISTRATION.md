# H6 — Hypothèse prospective verrouillée : 30 vs 25 features

**Statut au 16 octobre 2026 : protocole et validateur déposés sur la branche d'étude. Aucune nouvelle prédiction, aucun modèle candidat P3, aucun score prospectif ne sont disponibles. P3 56 jours demeure en pause.**

## Hypothèse unique

Sur BTCUSDT, ETHUSDT, SOLUSDT et XRPUSDT, le modèle H6 **25 features** (les 30 `FEATURE_COLUMNS` sans les cinq `candle_shape`) pourrait améliorer la décision directionnelle, comparé au modèle identique utilisant les **30 features**. H2 conserve son protocole à 30 features ; aucune autre suppression n'est envisagée dans cette expérience.

Colonnes supprimées exactement :

- `body_relative`
- `upper_wick_relative`
- `lower_wick_relative`
- `close_location`
- `range_pct`

**Provenance de l'hypothèse (checks historiques déjà inspectés, non indépendants)** : retrait des cinq +0,358 point d'uplift H6, 25/35 folds positifs, +0,000193 en log loss, +0,192 point de couverture. Par actif : BTC +0,307 (20/35), ETH +0,254 (21/35), SOL +0,356 (21/35), XRP +0,534 (22/35). Aucun de ces chiffres n'est une mesure de performance future.

## Calendrier et gel préalable

- **Début (inclus)** : `2026-10-17T00:00:00Z`.
- **Fin (exclue)** : `2026-11-14T00:00:00Z` (28 jours UTC).
- M5 clôturées, deux jeux de prévisions sur les mêmes barres et quatre actifs.
- Étiquette H6 définie selon la cible actuelle, maturité **30 min** après la clôture M5. Une évaluation complète ne peut avoir lieu avant le 14 novembre 2026 à 00:30 UTC.
- **IMPORTANT : avant le premier instant prospective**, entraîner les deux modèles avec les mêmes partitions antérieures, mêmes hyperparamètres LightGBM 3 classes, entraînement 120 j, arrêt anticipé 7 j, calibrage **uniquement des marges, sans labels** sur les 3 j de tuning précédents, `alpha=1.0`, cible de couverture `0.40`, seed `42`, au plus `150` estimateurs. Exclure les 4 derniers jours de check déjà inspectés de la sélection ; aucune consultation OOS/final 90 j ou ADA. Purger la maturité des labels H12 à toutes les frontières. Choisir une fenêtre antérieure à la date de début et identifier clairement ses bornes.
- Archiver les modèles sérialisés en lecture seule avec SHA-256 distinct, colonnes ordonnées, paramètres, priors d'entraînement, seuil de marge préfigé, identité/fingerprint des données et preuve de l'heure de gel. Aucun choix de candidat, de seuil, de date ou d'hyperparamètre après examen des résultats.
- La branche actuelle contient **uniquement le validateur hors-ligne**, pas le gel/collecte de deux modèles prospectifs. L'outil ne peut pas produire cette preuve lui-même.
- Si le gel ou la collecte préalable n'a pas eu lieu avant `START`, **ne pas effectuer de rejeu post hoc sous l'étiquette prospective** : définir par un nouvel amendement publié **avant** une nouvelle fenêtre future. Ne pas changer les dates dans le protocole v1 après coup.

## Journal de prédictions et schémas attendus

Un CSV `forecasts.csv` comporte pour chaque observation et chaque variante les colonnes :
`symbol,bar_close_at,predicted_at_utc,variant,p_down,p_neutral,p_up`.

Valeurs `variant` : `baseline_30` et `no_candle_shape_25`. Prédictions effectuées au moment de la clôture de la bougie et **avant** la maturité du label. Journal externe append-only / horodaté indispensable pour attester cette antériorité (CSV mutable et dates déclarées ne sont PAS une preuve).

Un CSV `labels.csv`, généré après maturité, contient `symbol,bar_close_at,label_mature_at,target_h6`, avec la cible existante dans `{0,1,2}`. L'évaluation refuse les clés manquantes entre variantes, doublons et moins de 95 % des 8 064 observations M5 théoriques par actif, ainsi que l'absence d'un jour UTC entier.

Un JSON `manifest.json` décrit le protocole, les deux modèles et leur gel : exemple **de structure uniquement**, à remplir par des valeurs réelles, jamais par des valeurs simulées.

```json
{
  "protocol": "h6_30_vs_25_forward_preregistered_v1",
  "horizon": 6,
  "coverage_target": 0.40,
  "uses_ada": false,
  "uses_fold_oos": false,
  "uses_reserved_final90": false,
  "models": {
    "baseline_30": {
      "feature_columns": ["REMPLACER_PAR_LES_30_COLONNES_ORDONNEES"],
      "model_sha256": "REMPLACER_PAR_SHA256_REEL",
      "threshold": null,
      "train_class_counts": null,
      "tuning_feasible": true,
      "model_frozen_at_utc": null,
      "last_training_label_mature_at_utc": null,
      "last_tuning_bar_close_at_utc": null
    },
    "no_candle_shape_25": {
      "feature_columns": ["REMPLACER_PAR_LES_25_COLONNES_ORDONNEES"],
      "model_sha256": "REMPLACER_PAR_SHA256_REEL",
      "threshold": null,
      "train_class_counts": null,
      "tuning_feasible": true,
      "model_frozen_at_utc": null,
      "last_training_label_mature_at_utc": null,
      "last_tuning_bar_close_at_utc": null
    }
  }
}
```

**Ne pas lancer le scorer avec cet exemple** : les contrôles échoueront intentionnellement tant que les modèles authentiques n'existent pas.

## Critères préenregistrés : tous exigés, puis revue humaine

1. Gain **pooled** d'uplift directionnel ≥ **+0,50 point** vs référence.
2. Limite inférieure du bootstrap descriptif 95 % des **28 différences journalières de gain** > 0 ; bootstrap déterministe 2 000 tirages de 28 jours avec remise, seed 42. Les journées partagées entre actifs sont gardées ensemble. **Ces jours restent potentiellement autocorrélés**, l'intervalle n'est pas un test confirmatoire universel.
3. Uplift moyen par actif strictement supérieur à la référence sur BTC, ETH, SOL et XRP.
4. Log loss brut du candidat **non détérioré**.
5. Écart de couverture réelle candidat - référence ≥ **−2 points**.
6. Couverture réelle **de chaque variante** comprise entre **30 % et 50 %**.

Si une seule porte échoue, aucun motif de promotion. Même si toutes réussissent, la décision reste **revue humaine uniquement**, jamais promotion automatique. Les critères sont des garde-fous préenregistrés et non un seuil de preuve statistique formel. Aucun calcul de PnL.

## Utilisation ultérieure (pas maintenant)

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h6_forward_candidate_evaluation `
  --models-manifest "E:\cryptoML-walkforward-results\h6_future_models_manifest.json" `
  --forecasts-csv "E:\cryptoML-walkforward-results\h6_future_forecasts.csv" `
  --labels-csv "E:\cryptoML-walkforward-results\h6_future_labels.csv" `
  --output "E:\cryptoML-walkforward-results\h6_forward_30_vs_25_v1.json"
```

L'outil refuse l'analyse avant la maturité entière de la fenêtre, refuse tout écrasement et inscrit les hachages SHA256 des fichiers d'entrée dans son rapport. Cela ne remplace PAS un système d'enregistrement des prédictions réellement antérieur à leur étiquette.

## Règles d'intégrité

Interdit : reprendre les validations P2 vues comme preuve prospective, consulter ADA P0, les folds OOS scellés ou les 90 derniers jours scellés, modifier P3, relancer le monitor 56 j, fusionner le candidat 25 features avec les modèles figés sans décision distincte et preuve nouvelle. L'outil nouveau n'entraîne aucun modèle et n'ingère aucune donnée propriétaire.
