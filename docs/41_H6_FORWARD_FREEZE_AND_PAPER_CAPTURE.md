# H6 30 vs 25 — Préparer le gel et la collecte (sans réactiver P3)

**IMPORTANT : trois programmes hors ligne, sans ordonnanceur, sans appel réseau et sans exécution de trading. Rien ne démarre automatiquement.** Les 90 derniers jours réservés, les tests OOS des folds et ADA restent intouchés, y compris pour le choix de modèles. Le moniteur P3 de 56 jours demeure suspendu.

Fichiers :

- `mlpipeline.h6_forward_freeze` : entraîne deux LightGBM H6 distincts sur le **dernier fold de développement** permis, déterminé sans consulter les résultats OOS, puis ajuste chacun son seuil sur les trois jours de tuning, sans jamais examiner le check. Refuse les seuils infaisables ou un gel postérieur au début du protocole.
- `mlpipeline.h6_forward_collect_once` : lit un **CSV de bougies M5 clôturées fourni par l'opérateur**, valide les quatre actifs, recalcule les 30 features causales avec le contexte BTC/ETH, puis infère les deux modèles et inscrit un fichier JSON par clôture M5, sans l'écraser. Refuse une clôture antérieure, non définitive, décalée, ou saisie plus de deux minutes après l'heure réelle de clôture.
- `mlpipeline.h6_forward_candidate_evaluation` : **plus tard**, calcule les critères finaux uniquement sur les prédictions préenregistrées et étiquettes post-maturité.

## À faire maintenant : valider les tests

Depuis `E:\cryptoML-walkforward-test`, synchroniser la branche dans le checkout distinct :

```powershell
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

## Gel séparé : seulement avant START si tests verts

**Date de début du protocole v1 : 17 octobre 2026 00:00 UTC = 02:00 heure de Paris (UTC+2).**

Le répertoire de sortie **ne doit pas exister**. Le chemin P0 est strictement lu, jamais modifié. Le bundle ne fait **pas** partie de P3.

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h6_forward_freeze `
  --data-root "E:\cryptoML\data" `
  --output-dir "E:\cryptoML-walkforward-results\h6_forward_frozen_30_25_v1" `
  --months 24 --n-estimators 150 --seed 42 --verify-hashes
```

Deux fichiers `.joblib` et un `manifest.json` doivent apparaître. Le rapport inclut les seuils de tuning, SHA256 modèles, empreinte des archives historiques et bornes du fold. **Préserver une copie immuable externe du bundle avant le début de la collecte**, horodatée par une source vérifiable ; les horloges locales et les SHA256 sans témoin externe ne prouvent pas l'antériorité.

Le programme s'arrête si les 40 % de couverture sont irréalisables sur le tuning de l'un des modèles. Dans ce cas, **ne pas choisir un autre fold en fonction de ces résultats** ; enregistrer l'échec et définir un nouveau protocole.

## Collecte future — uniquement quand quatre flux externes M5 définitifs sont déjà disponibles

Aucun flux de bougies externe automatique n'est intégré. Le CSV doit venir d'une **source indépendante** et contenir pour chacun des quatre actifs au moins 100 barres M5, y compris une **même dernière clôture UTC**. Colonnes obligatoires :

`symbol,bar_open_at,bar_close_at,open,high,low,close,volume_base,volume_quote,trade_count,taker_buy_volume_base,is_final`

`bar_close_at` est le temps de clôture exclusif (open+5 min). Les deux temps portent explicitement `Z` ou un décalage `+00:00`. Les quatre dernières bougies doivent être `is_final=True`. Aucun ADA. Exemple de commande **uniquement si un véritable CSV récent existe et que la clôture est dans le créneau autorisé** :

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h6_forward_collect_once `
  --frozen-bundle "E:\cryptoML-walkforward-results\h6_forward_frozen_30_25_v1" `
  --closed-m5-snapshot-csv "E:\cryptoML-walkforward-results\incoming_final_m5_snapshot.csv" `
  --output-dir "E:\cryptoML-walkforward-results\h6_forward_paper_logs_v1"
```

**ATTENTION : ne pas créer un CSV artificiel ou issu des périodes P0 réservées pour faire passer le collecteur.** La commande ne doit être exécutée que sur un flux réellement recueilli et archivé à l'heure, sans reconstitution. Comme il n'y a ni téléchargement automatique ni boucle d'ordonnancement, un lancement unique enregistre **une seule clôture M5**. Cette version n'est pas un dispositif complet de collecte 28 jours ; il faudrait mettre en place un transport de données et un ordonnanceur distincts puis les auditer, sans réveiller P3.

### Limites et décision go/no-go

Si aucune collecte authentique et continue n'est prête avant `2026-10-17T00:00:00Z`, **la fenêtre v1 n'est pas utilisable comme validation prospective**. Ne pas antidater les artefacts ni lancer un backfill ; publier une nouvelle fenêtre v2 avant son début puis adapter explicitement les contrôles. Les outils v1, volontairement stricts, refuseront tout décalage hors de leur fenêtre.

L'évaluation future demeure distincte de tout bénéfice de trading ou toute décision de déploiement. Voir `docs/40_H6_30_VS_25_FORWARD_PREREGISTRATION.md`.
