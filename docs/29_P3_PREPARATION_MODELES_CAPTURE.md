# CryptoML P3 — Préparation des modèles et capture M5

**État : code de préparation + capture ponctuelle ajouté, NON ACTIVÉ, NON VALIDÉ de bout en bout.** Le préenregistrement existant `prospective_h2_h6_p3` demeure inchangé ; si la chaîne n'est pas opérationnelle au 16 octobre 2026 à 00:00 UTC, inscrire une nouvelle version/dossier futur, ne jamais antidater.

## Protection historique

`mlpipeline.prospective_model_pack` lit explicitement les partitions P0 **décembre 2025 à juin 2026 seulement**, et seulement BTC/ETH/SOL/XRP. Il interdit de charger juillet–septembre (réserve historique déjà consultée) et ADA. H12 purge, 120 jours train, 7 jours early stop, 3 jours tuning pour seuil de sélection 40% à 12–17 UTC et toutes les heures. H2 et H6 utilisent les mêmes observations, mais ont des labels différents. Il sauvegarde deux objets LightGBM de confiance, les priors train et les seuils sans jamais noter une période prospective.

## Windows — synchronisation et tests

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q
```

Après tests verts, préparation hors des dossiers gelés (avec validation préalable du contenu historique) :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_model_pack `
  --data-root "E:\cryptoML\data" `
  --protocol-root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3" `
  --verify-hashes `
  --output "E:\cryptoML-walkforward-results\p3_model_packs_pre_july"
```

L'exécution génère `h2_model.joblib`, `h6_model.joblib`, `threshold_policy.json`. Le dossier ne peut pas être réécrit. Les artefacts joblib doivent provenir uniquement de cette procédure de confiance : ne JAMAIS désérialiser un fichier joblib téléchargé ou inconnu.

## Capture live (à tester avant activation)

`mlpipeline.prospective_live_capture` charge les deux artefacts de confiance seulement après vérification de `activation.json` et des hashes, télécharge les dernières bougies M5 clôturées depuis l'API publique USD-M Binance, construit les 30 features, compare les horodatages de clôture sur les quatre actifs et calcule les sorties H2/H6. Une requête tardive (>270 s après la clôture commune) échoue ; aucune prédiction n'est reconstituée rétroactivement.

Les résultats sont ajoutés dans un journal local `predictions.jsonl`, avec chaînage SHA256 et fsync. Ce journal ne prouve PAS l'antériorité par rapport au marché : un opérateur pourrait encore reconstituer entièrement le fichier. Une preuve externe de temps et un stockage WORM/append-only distant sont requis. La capture ne réentraîne pas, ne produit pas de performance et n'envoie pas d'ordres.

Après construction/vérification de vrais packs et d'une infrastructure de capture sûre, la commande d'activation existante est :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_protocol activate `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3" `
  --h2-model "E:\cryptoML-walkforward-results\p3_model_packs_pre_july\h2_model.joblib" `
  --h6-model "E:\cryptoML-walkforward-results\p3_model_packs_pre_july\h6_model.joblib" `
  --threshold-file "E:\cryptoML-walkforward-results\p3_model_packs_pre_july\threshold_policy.json"
```

**NE PAS appeler activate à ce stade** : le journal local, le planificateur M5 fiable et l'ancrage d'horodatage indépendant ne sont pas entièrement implémentés ni vérifiés.

La commande de capture ponctuelle (pour référence après véritable activation) serait :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_live_capture `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3"
```

Aucun service ou ordonnanceur n'est installé par le code ; lancer une commande une seule fois ne permet pas de collecter 56 jours de données. Le futur planificateur devra garantir un seul processus et une seule capture par M5, la synchronisation fiable de l'horloge, l'archivage externe et la surveillance des pannes. Les bougies manquantes doivent rester manquantes, sans backfill.

Enfin, le composant de scoring prospectif, à déclencher uniquement après maturité de tous les labels et contrôle des preuves, n'est pas encore disponible. Ne jamais annoncer une réussite sur la seule base d'un protocole, d'une activation ou de fichiers historiques.