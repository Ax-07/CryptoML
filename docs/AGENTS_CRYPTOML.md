# Instructions de développement — CryptoML

À utiliser avec un assistant de code depuis VS Code. Documentation de référence : `README_CRYPTOML.md` et `docs/`.

## Principes

- **Respecter le dépôt existant** : inspecter l'arborescence, les conventions, les versions et les composants avant de modifier quoi que ce soit. Ne pas recréer Next.js ni écraser des fichiers du projet.
- **Construire par étapes verticales vérifiables** : ingestion → features/labels → première baseline et LightGBM → évaluation → API → UI.
- **Ne pas inventer des chiffres** : les jeux de démonstration sont explicitement `demo=true` et séparés des résultats réels.
- **UTC partout**, `bar_open_at` / `bar_close_at` clairement distingués ; prévision émise seulement après clôture confirmée.
- **Anti-fuite prioritaire** : calculs par symbole, features historiques, labels hors features, test inédit immuable, purge des échantillons chevauchant les frontières.
- **Pas de trading live** : aucune création d'ordres ni demande de credentials privés d'exchange. Futures = **type de données de marché**, pas exécution.
- **Code lisible et typé** : Pydantic côté API, TypeScript côté frontend, tests pour toute transformation temporelle.
- **Conserver les versions** : `dataset_id`, `feature_set_version`, `label_config_hash`, `model_version`, `run_id` dans les résultats.
- **Pas de promesses de rentabilité** et pas d'interprétation de `prob_up` comme probabilité de gagner de l'argent.
- Les champs et routes de `contracts/openapi.yaml` constituent une **proposition de contrat** : ajuster de façon explicite et documenter tout changement.

## Ordre recommandé pour une première livraison

1. Ajouter `backend/` à la structure existante ; config et tests.
2. Implémenter `ingestion` des archives Binance USD-M + incrément REST et contrôle de clôture.
3. Produire partitions Parquet validées.
4. Construire 10–20 features sans fuite, labels H=6, évaluation baseline.
5. Train LightGBM global, splitting temporel et rapport OOS.
6. Endpoints `candles`, `models`, `evaluations`, `predictions`, `replay`.
7. Connecter la page Qualité prédictive et Lightweight Charts.

## Définition de terminé

- La fonctionnalité fonctionne sur un vrai dataset et sur un petit fixture déterministe.
- Tests unitaires et tests anti-fuite passent.
- Pas de comparaison entraînement/test contaminée.
- Les erreurs de chargement et les données absentes sont traitées explicitement.
- L'API, le contrat et la documentation sont mis à jour ensemble.
