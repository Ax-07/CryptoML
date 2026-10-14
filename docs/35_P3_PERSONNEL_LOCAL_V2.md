# P3 personnel V2 — collecte locale sans Certigna

**Pourquoi cette V2 :** expérience uniquement personnelle. L'ancienne inscription `prospective_h2_h6_p3` (V1, 16 octobre) doit rester intacte et inactive. La collecte locale ne prétend **pas** à une antériorité vérifiée indépendamment. Le protocole scientifique (56 jours, H2/H6, UTC 12–17, journée entière, critères finaux, 4 actifs, ADA exclu) reste inchangé.

## Sécurité conservée

- Packs SHA256 exacts : H2 `6DC2FC0A...`, H6 `D2745DE3...`, seuils `3DD655DF...`, vérifiés **avant** chargement `joblib`.
- Nouvel identifiant `cryptoml_p3_h2_h6_personal_local_v2` et NOUVEAU dossier exclusivement.
- Départ explicitement choisi à minuit UTC, au moins 48 heures après inscription. Aucune modification du JSON gelé.
- Un créneau M5 tenté au plus une fois ; claims persistants avant toute prédiction, journal SHA256 local, verrou Windows ; aucun backfill.
- Résultats stockés avec les politiques **12-17_UTC** et **all_day_UTC** pour chaque horizon et chaque actif.
- Aucune vérification des résultats du marché jusqu'à 56 jours + 60 min ; ni trading ni re-tuning.
- **Limites :** horloge locale Windows, coupures réseau, extinction PC et altération volontaire des journaux ne sont pas couverts par une preuve indépendante. Les 56 jours nécessitent PC allumé et suivi des lacunes.

## Procédure Windows (après tests verts)

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Étape inscription, sans lancement de collecte.** Exemple date à adapter en fonction de l'heure réelle et du minimum de 48 h : `2026-10-18T00:00:00Z`. Ne pas créer si ce début est devenu trop proche ou passé.

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_personal_protocol register `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_personal_v2" `
  --start-utc "2026-10-18T00:00:00Z"
```

Vérifier l'inscription :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_personal_protocol verify `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_personal_v2"
```

**Étape de gel**, à effectuer exclusivement après les tests et validation des SHA256 :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_personal_protocol activate `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_personal_v2" `
  --pack-root "E:\cryptoML-walkforward-results\p3_model_packs_pre_july"
```

**Ne pas activer avant vérification des tests, de l'heure Windows et des sauvegardes.** La commande `activate` gèle les hashes dans la nouvelle inscription ; elle ne génère pas encore de prévisions.

**Démarrage du collecteur**, uniquement après gel vérifié, et sous terminal ou supervision Windows dédiée :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.prospective_personal_scheduler `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_personal_v2"
```

Ce processus ne s'installe pas en tant que service Windows et **ne survit pas à la fermeture du terminal**. Éviter veille, arrêts Windows et changements manuels d'horloge. La supervision et la sauvegarde sur un autre support doivent être configurées et éprouvées avant que la collecte puisse être tenue pour fiable. Ne pas lancer avec un faux prestataire. Les scripts V1 nécessitant ancrage tiers restent inchangés.

## Backup personnel recommandé

Copier `protocol.json`, `protocol.sha256`, `activation.json`, `activation.sha256`, `attempts.jsonl`, `predictions.jsonl` régulièrement (idéalement en snapshots immuables datés) vers un autre support ou cloud personnel. Utiliser une copie cohérente en dehors de l'écriture ; ne jamais réécrire ou réinjecter le journal de capture pour combler un trou. Une sauvegarde simple accroît la robustesse pratique, mais ne prouve pas indépendamment l'antériorité.

Aucune commande ci-dessus n'est exécutée par ChatGPT sur le PC de l'opérateur.
