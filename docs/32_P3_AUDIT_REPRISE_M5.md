# P3 — Audit opérationnel avant tout démarrage

**Statut : code de reprise et garde anti-backfill ajouté, non testé sur le Windows opérateur. P3 reste INACTIF.**

Les 114 tests précédents validaient l'inférence réelle en lecture seule. Ils ne prouvent pas la fiabilité sur 56 jours.

## Reprise après panne

- `mlpipeline.prospective_attempts.claim_attempt` inscrit **avant toute requête** une tentative M5 dans `attempts.jsonl` (fsync, chaînage SHA256, verrou interprocessus).
- Un second passage ou un redémarrage refuse la même clôture. Une tentative réseau échouée est laissée **manquante** ; aucun retour en arrière.
- Les bougies Binance doivent avoir exactement la clôture réclamée par le planificateur. Un retard de publication n'autorise pas de prendre la bougie précédente.
- `predictions.jsonl` conserve, lui, les seules prévisions avec reçu externe valide ; `attempts.jsonl` est opérationnel local et ne sert **jamais** de preuve prospective indépendante.
- Les fichiers locaux peuvent être altérés par un opérateur ayant accès au disque. Le SHA local n'est pas une infrastructure WORM ni une garantie d'horloge.

## Vérification Windows

```powershell
cd E:\cryptoML-walkforward-test
git status
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**Attendu : 117 tests** si les 3 nouveaux tests sont les seuls ajouts et qu'aucune régression n'existe.

## Conditions strictes encore non remplies

1. Contrat HTTPS d'ancrage Ed25519 testé contre un tiers réellement indépendant avec horloge fiable. **Le serveur tiers et le stockage append-only ne sont pas fournis**.
2. Preuve que ce serveur conserve les digests et accepte de les restituer pour audit ; vérifier son identité, sa politique de rétention et sa résistance à la réécriture. Une signature seule ne suffit pas.
3. Supervision Windows du processus (redémarrage automatique, contrôle de l'heure, alertes, journaux hors du disque local, accès réseau et tolérance aux pannes) sur un protocole de staging **différent** de P3.
4. Test intégral sur staging : vrais packs immuables, pré-enregistrement futur, pack activé seulement dans ce staging, signature indépendante obtenue avant la prochaine clôture M5 et récupération de la preuve, sans regarder les labels. Ne jamais activer la fenêtre actuelle juste pour tester.
5. Audit méthodologique de la procédure finale avant toute période prospective ; scoring seulement à l'expiration de la fenêtre et de l'embargo H6/H12.

## Calendrier

La registration `prospective_h2_h6_p3` commence **2026-10-16T00:00:00Z**. Si les garanties ne sont pas établies à cette échéance, laisser cette inscription non active. Inscrire ensuite une **nouvelle version** sous un nouveau dossier pour des dates futures ; ne jamais antidater des prédictions, réécrire le protocole initial ni classer une preuve synthétique comme preuve de production.

Aucun ordre de trading n'est émis. Aucune donnée ADA, OOS anciennement réservée ni résultat futur n'est évalué.
