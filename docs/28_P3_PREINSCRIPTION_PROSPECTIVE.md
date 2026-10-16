# CryptoML P3 — Préenregistrement prospectif H2 contre H6

**État : protocole et garde de gel préparés ; aucune collecte ni validation prospective n'est encore active.**

Les 36 anciennes fenêtres, les checks et les 90 jours finaux déjà consultés ne constituent pas de nouvelles confirmations. Un test prospectif requiert des prédictions enregistrées avant maturité des labels avec un horodatage vérifiable ; ce module ne réalise pas encore la collecte ni le scoring.

## Hypothèse figée

- H2=10 minutes contre H6=30 minutes ; 30 features, LightGBM CPU 3 classes, alpha 1.0, labels seuil 0.5.
- Cohorte principale 12 h–17 h 59 UTC, cible de couverture 40 % sur scores antérieurs (seuils gelés avant collecte). Analyse secondaire sur 24 heures.
- Durée préenregistrée : 56 jours consécutifs / 8 semaines non chevauchantes. Score uniquement à la fin + 60 minutes.
- Critères descriptifs préspécifiés : moyenne des écarts d'uplift H2−H6 strictement positive, au moins 6/8 semaines positives, >=95 % de couverture des timestamps M5 attendus, couverture directionnelle de 35–45 % sur chaque horizon, différence réalisée <=5 points.
- Pas de changement des modèles, features, seuils, cohortes après activation ; aucune promesse de rentabilité. Si une condition de validité échoue, la conclusion est non concluante.
- Models / seuils pré-entraînés devront exclure la fenêtre finale historique déjà consultée, pour éviter de recycler ses labels comme confirmation. ADA reste exclue.
- La cohorte 12–17 UTC a été choisie après exploration : même un résultat positif n'est pas une preuve indépendante de rendement.

## PowerShell : phase A uniquement — préenregistrement

Dans le worktree propre :

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_protocol register `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3"

& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_protocol verify `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3"
```

Le dossier créé contient protocol.json et protocol.sha256. Une deuxième inscription dans le même dossier est refusée. Si le protocole doit changer ou si la collecte n'est pas prête à la date de départ : **nouveau protocole/dossier versionné**, jamais de redatation rétroactive.

## Phase B — non implémentée : préparer les vrais modèles et seuils

Il faut encore entraîner/geler deux artefacts H2/H6 et calibrer les seuils 40 % sur données antérieures. Le module offre une commande d'activation qui vérifie les métadonnées déclarées et les empreintes SHA256 des fichiers, mais ne prouve pas que leur entraînement est honnête. Ne pas appeler activate avec des fichiers fictifs.

## Phase C — non implémentée : collecte et évaluation

Créer une chaîne de prédictions M5 réellement émises en temps réel, écrites en registre append-only avec preuve d'horodatage externe, avant fermeture des horizons futurs. Interdire les ajouts rétroactifs. Après 56 jours + maturité H6, un composant séparé pourra seulement alors associer les labels matures et calculer les métriques.

Les hashes locaux ne suffisent pas à prouver qu'une prédiction précède les événements. Cette infrastructure et ses tests doivent exister avant l'activation pour qualifier la collecte de prospective.