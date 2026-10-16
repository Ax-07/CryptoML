# P3 — Vérification en lecture seule des vrais packs H2/H6

**Aucun modèle n'est réentraîné, aucune prédiction n'est enregistrée, aucun résultat n'est noté.** Ce contrôle est séparé du préenregistrement, de l'activation et de la collecte.

## Empreintes de référence consignées par l'opérateur

- H2 : `6DC2FC0A176BF2D719C7B468626BDE2CEBF16848D345E86268C4C1DC88A0F287`
- H6 : `D2745DE33F21D8B1E64C7B2AD850F96A418E506352E03F92F394D7C041B9F097`
- Seuils : `3DD655DF3823FE9DDBF7BA6B1E721ED67981B90651C1C2FF16A6459F420A065A`

Le module `mlpipeline.prospective_preflight` rejette les fichiers avec un autre SHA256 **avant** de charger les objets joblib. Le hachage ne garantit pas à lui seul l'origine du programme, mais rend observable tout changement postérieur à ces références.

## Exécution dans le venv P3 isolé

Depuis PowerShell :

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Puis valider les artefacts réels sans requête réseau ni nouvelle prédiction :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_preflight `
 --protocol-root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3" `
 --pack-root "E:\cryptoML-walkforward-results\p3_model_packs_pre_july"
```

Enfin, pour un test d'inférence des packs réels **sur les dernières bougies Binance clôturées** (aucun fichier de prédictions n'est créé), reprendre la commande précédente avec `--live-check`. Ce test consomme seulement les données de marché publiques récentes, sans étiquette future ni performance. Il faut pouvoir joindre `https://fapi.binance.com` depuis Windows.

L'inférence de diagnostic calcule les deux politiques H2/H6, vérifie que les probabilités sont finies et les quatre actifs présents, mais **n'affiche pas leurs décisions** et ne conserve pas leurs prédictions.

## Point d'arrêt avant P3

Aucun service HTTPS d'horodatage réellement indépendant n'est raccordé au code. Le contrat Ed25519 exige un opérateur tiers, un ancrage de digest persistant, et une clé publique auditable. L'ordonnanceur local ne garantit pas seul une conservation WORM externe ni une supervision 56 jours.

Il faut donc garder `prospective_h2_h6_p3` **non actif**. Départ enregistré : **2026-10-16T00:00:00Z**. S'il n'est pas opérationnel avant le départ, conserver cette inscription comme tentative non démarrée, ne pas la modifier ou la réutiliser : enregistrer un NOUVEAU dossier avec période prospective future une fois l'infrastructure vérifiée. La date d'entraînement du pack n'est pas changée.
