# CryptoML P3 — Double politique, M5 et horodatage externe (intégration en cours)

**Statut : code et tests ajoutés sur la branche de recherche, activation interdite tant qu'un service indépendant n'est pas mis en place et audité.** Le préenregistrement du 16 octobre 2026 n'est pas modifié.

## Double politique

Chaque bougie et chaque actif conservent, pour **chacun** des modèles H2 et H6 :
- la distribution brute DOWN/NEUTRAL/UP et la marge de décision ;
- la décision avec seuil `12-17_UTC` ;
- la décision avec seuil `all_day_UTC` ;
- l'appartenance ou non au créneau principal.

**Important :** l'analyse secondaire journée entière inclut le créneau principal ; il ne faut jamais employer le seuil principal en remplacement du seuil journée entière.

## Horodatage externe obligatoire (pas de fallback)

`mlpipeline.prospective_anchor.TrustedAnchor` exige une URL HTTPS distante configurée par l'opérateur et une clé publique **Ed25519** hors des modèles. Le serveur externe (NON fourni avec ce dépôt) doit conserver de façon append-only le digest de l'enregistrement, la date UTC d'émission et un identifiant stable de service, puis retourner une signature Ed25519 sur un JSON canonique.

Format de la requête HTTPS POST :

```json
{"activation_sha256":"…","bar_close_utc":"2026-10-16T00:05:00Z","digest_sha256":"…","protocol_sha256":"…"}
```

Réponse attendue :

```json
{"signed":{"digest_sha256":"…","issued_at_utc":"2026-10-16T00:05:17Z","service_id":"INDEPENDENT_PROVIDER"},"signature_base64":"…"}
```

**Le tiers doit être réellement indépendant, disposer d'une horloge fiable et assurer la rétention vérifiable des preuves.** Ce projet ne fournit pas de tel prestataire ; une signature d'un faux serveur contrôlé par le même processus ne démontre pas l'antériorité des prédictions. Les signatures sont vérifiées, mais le tiers doit encore être sélectionné, audité et testé. La capture refuse d'écrire sans reçu valide ou si le délai après clôture dépasse 270 secondes. Aucun mode local ou « test » ne peut contourner cette exigence dans la commande de capture.

## Écrivain unique et ordonnanceur

`prospective_journal` verrouille un fichier stable sur Windows (msvcrt) ou POSIX (flock), interdit doublons, horodatages décroissants et lignes tronquées. Il vérifie l'intégrité du chaînage et les signatures externes à chaque extension du journal.

`prospective_scheduler` tente une seule capture M5 à partir de 15 secondes après la fermeture UTC, sans rejouer une bougie manquante. Il n'installe aucun service Windows et ne se lance pas depuis ChatGPT. Un système d'exploitation/superviseur externe doit maintenir le processus, le surveiller et horodater les interruptions. La collecte sans surveillance entraînerait une perte de données difficile à interpréter.

## Tests et dépendance

Depuis `E:\cryptoML-walkforward-test\backend`, après mise à jour de la branche :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pip install "cryptography>=43,<48"
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m pytest -q
```

**ATTENTION :** ce pip modifie uniquement le venv existant `E:\cryptoML\backend\.venv` commun aux deux worktrees. Ne l'effectuer que si l'environnement partagé peut être mis à jour, sinon créer un venv isolé dans le worktree expérimental.

Les commandes réservées à **une future activation testée**, après raccordement du vrai serveur d'horodatage :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.prospective_scheduler `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3" `
  --anchor-url "https://SERVICE-INDEPENDANT/attest" `
  --anchor-public-key "E:\CHEMIN\cle-publique-ed25519.pem"
```

**Ne pas exécuter cette commande avec des paramètres fictifs.** Si le préenregistrement du 16 octobre n'est pas opérationnel avant minuit UTC, il reste non démarré ; produire une nouvelle version datée à l'avance. Ne pas écrire rétroactivement.

## Points non terminés

- Aucun prestataire d'horodatage indépendant réellement configuré, aucune preuve de stockage WORM attestée.
- Aucun processus Windows/service supervisé installé ou validé en conditions de réseau réelles.
- Aucun outil de scoring prospectif final vérifié et gelé, à développer **sans** regarder les résultats prématurément.
- Modèles historiques préparés mais **pas activés** ; aucun ordre, trading ni résultat futur.
