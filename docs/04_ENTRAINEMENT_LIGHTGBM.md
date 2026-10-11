# 04 — Entraînement LightGBM multi-crypto

## Modèles

- Trois `LGBMClassifier(objective='multiclass')` globaux distincts : `h3`, `h6`, `h12`.
- Labels de classes `{0: DOWN, 1: NEUTRAL, 2: UP}`.
- Même schéma de features sur tous les actifs ; `symbol` absent de `X` en V1, conservé pour métriques et splits.
- Baselines : classe majoritaire train, persistance du signe du rendement récent avec gestion explicite du neutre, régression logistique régularisée.

## Pipeline d'un run

1. Référencer un `dataset_id` gelé, `feature_schema_version`, `label_config_id` et univers de symboles.
2. Extraire toutes les samples éligibles et trier chronologiquement.
3. Construire un split temporel avec paires train/validation/test définies **avant** entraînement.
4. Purger les samples dont les `label_mature_at` tombent dans la période suivante ; prévoir aussi un embargo si nécessaire.
5. Calculer les statistiques de traitement manquant/normalisation strictement sur train. LightGBM ne requiert pas de scaling systématique.
6. Entraîner les modèles par horizon ; utiliser `early_stopping` sur la validation propre au fold.
7. Générer prédictions OOS + métriques + classes + SHAP/importance des features + manifests.
8. Sauvegarder modèle, seed, hyperparamètres, version environnement, commit Git et métriques dans MLflow.
9. **Ne jamais améliorer les hyperparamètres en fonction du score du test final.**

## Paramètres de départ (à ajuster sur validation uniquement)

```yaml
model: lightgbm_multiclass
objective: multiclass
n_estimators: 800
learning_rate: 0.03
num_leaves: 31
max_depth: -1
min_child_samples: 150
subsample: 0.8
subsample_freq: 1
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0
random_state: 42
early_stopping_rounds: 50
```

Ces valeurs sont **hypothèses de départ**, pas hyperparamètres validés. Tester des modèles plus simples et le coût en mémoire. Ne pas utiliser le dernier fold comme validation après avoir vu son résultat final.

## Équilibrage multi-crypto

Les 20–30 paires peuvent avoir des durées d'historique différentes et des volatilités très variées. Évaluer une pondération symboles/périodes calculée sur train, sans pénaliser arbitrairement les nouvelles paires. Reporter à la fois **métrique agrégée** et **métriques par symbole**.

## Reproductibilité

Chaque run doit persister :

```text
run_id, horizon_bars, universe_id, dataset_id,
feature_schema_version, label_config_id, splits_manifest_id,
model_family, hyperparameters, random_state,
training_started_at, training_finished_at, git_commit,
artifacts_path, run_status
```

Les prédictions OOS stockées sont associées au **run/fold** qui les a produites et ne doivent jamais être recalculées a posteriori avec un modèle plus récent puis présentées comme historiques.

## Statuts job

`QUEUED → RUNNING → EVALUATING → COMPLETED`, ou `FAILED`, `CANCELLED`. Le frontend suit la progression, les étapes et les logs filtrés ; un échec doit garder message technique et trace sans divulguer de secrets.
