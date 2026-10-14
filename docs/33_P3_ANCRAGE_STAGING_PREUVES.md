# P3 — Staging d'horodatage indépendant et récupération de preuve

**État : contrat d'intégration ajouté, AUCUN fournisseur configuré ni testé en réel.** P3 reste non actif. Ne pas utiliser les faux noms DNS du test comme prestataires.

## Contrat réseau minimal

Un prestataire externe, réellement indépendant de la machine qui génère les prévisions, doit fournir :
- HTTPS POST avec JSON canonique du digest SHA256 de la prévision ;
- reçu signé Ed25519 contenant `digest_sha256`, `issued_at_utc`, `service_id` ;
- HTTPS GET indépendant `/proof/{digest}` restituant **le même reçu** immédiatement ;
- rétention append-only vérifiable et récupération future pour audit, politique de temps UTC documentée ;
- preuve de sa gouvernance et de son horloge indépendantes. Une simple signature de notre propre serveur ne suffit pas.

Le client refuse l'écriture si la signature n'est pas valable, si le digest diffère, si l'émission est postérieure de 270 s ou plus à la clôture, si le GET échoue, ou si le reçu récupéré diffère. Un serveur HTTPS local ou une adresse IP littérale sont rejetés, sans garantie que le DNS choisi soit réellement externe.

## Test de staging, sans prévisions et sans activation

Dans le worktree isolé :

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Pour le vrai prestataire, **après contrat, vérification d'indépendance et clé publique obtenue par canal sûr**, utiliser :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_anchor_probe `
  --anchor-url "https://URL-REELLE-DU-TIERS/attest" `
  --anchor-lookup-url "https://URL-REELLE-DU-TIERS/proof/{digest}" `
  --anchor-public-key "E:\chemin\cle-publique-du-tiers.pem"
```

**Ne pas exécuter avec URLs fictives.** La commande génère un nonce synthétique, un digest, teste POST + GET + Ed25519, ne charge pas les packs, ne crée ni journal de prévisions ni modèle, ne modifie pas la préinscription.

La validation de l'immuabilité réelle et de la rétention sur 56 jours nécessite un audit indépendant de l'infrastructure distante ; le test immédiat POST+GET ne démontre pas cette propriété.

## Conditions de départ

Le début du protocole initial est **2026-10-16T00:00:00Z**. Ne pas utiliser un service non vérifié pour contourner cette date. Si le fournisseur et le superviseur Windows ne sont pas validés avant cette échéance, enregistrer **un nouveau dossier** avec un nouveau début futur, sans modifier le protocole original.

Les modèles et seuils H2/H6 déjà calculés demeurent figés, sans nouveau tuning, OOS, ADA ni ordres de trading.
