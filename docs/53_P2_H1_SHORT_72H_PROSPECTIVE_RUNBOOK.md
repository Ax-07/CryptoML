# P2 — Confirmation H1 très courte (72 h), pas de suivi permanent

**Statut** : outil technique prêt pour tests ; non exécuté ici. Date du protocole : 19 octobre 2026.

## Périmètre strict

Une seule fenêtre de **72 heures UTC** à partir d'un départ futur automatiquement fixé au lancement ; **arrêt automatique**. Quatre actifs déjà étudiés seulement : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT. Bougies Binance USD-M perpetual **M5 uniquement**, lecture des **données publiques REST**, sans API privée, sans ordre ni PnL.

Les trois modèles sont gelés une fois au démarrage à partir du **train de 120 jours et de l'early stop 7 jours du dernier fold historique P2 original (fold 35)**, purge originale H12. Cela n'est **pas** une reprise des poids des modèles expérimentaux des 36 folds ; ce sont **trois nouveaux modèles fixes de même spécification**, entraînés exclusivement sur train+early historiquement autorisés, puis gelés. **Les anciens checks P2, les tests OOS du fold, ADA et le holdout 90 jours ne sont jamais exploités pour l'entraînement.** Le modèle 30 features gelé de production ou P3 n'est ni lu ni modifié. Trois LightGBM multiclasses CPU identiques en hyperparamètres (max 150 itérations, seed 42) :
1. Deux EMA (\`close/EMA9-1\`, \`EMA9/EMA26-1\`).
2. Deux EMA et \`close_location\`.
3. Trente features historiques \`features_v1\`, dans l'ordre gelé.

Cible primaire : **signe brut de la prochaine bougie M5**, hors future variation nulle. Les modèles demeurent entraînés sur H1 à **trois classes** par la règle volatilité 0.5× et leur signe forcé compare P(UP) et P(DOWN), exactement comme en exploratoire. Références figées : toujours DOWN et réversion du rendement de la dernière bougie. Les résultats sont appariés sur chaque même bougie effectivement enregistrée. Publication séparée par actif, couverture des bougies attendues et nombre d'observations, bilan après maturation totale.

## Frontières techniques

- **Pas de prédictions antidatées** : \`freeze\` scelle le fichier des trois modèles avec SHA256 et planifie un début futur (prochain multiple de 5 minutes +10 minutes). \`observe --follow\` reste au premier plan, démarre les requêtes à la clôture de chaque M5 et écrit le fichier \`predictions.jsonl\` avec \`fsync\` **avant que la prochaine bougie M5 ne clôture**. Aucune publication de prédiction pour un prix futur déjà connu. Les premières 4h15 de données publiques fraîches servent uniquement à initialiser les 30 features + contexte BTC/ETH (une EMA26 seule en requerrait moins).
- **Aucun rattrapage** : une interruption Internet, des données incomplètes, un processus arrêté ou une clôture ratée ne sont pas convertis en prédictions fictives ; les observations manquées restent manquantes. Le programme peut être relancé dans la même fenêtre après avoir vérifié le lock, mais **ne prolonge jamais** les 72 heures.
- \`evaluate\` n'est autorisé **qu'après 72 h + 6 min**, vérifie que chaque observation conservée précède la réalisation H1, recoupe les clôtures d'évaluation avec les données publiques, puis écrit un **seul** \`final_report.json\`.
- **Lecture seule** P0 historique ; aucune écriture dans \`E:\cryptoML\data\`, \`E:\cryptoML\backend\`, l'ADA, les périodes OOS réservées, P3 ou les poids et schémas de production. Les données REST récentes sont traitées en mémoire ; seules les prédictions et méta-données de l'expérimentation sont enregistrées sous \`E:\cryptoML-walkforward-results\`.
- Pas d'appels à des outils de trading, pas d'exécution de stratégie ni décision automatique sur les résultats. **Une comparaison H1 à peine 72h n'a aucune puissance pour établir la stabilité long terme, la rentabilité ou l'indépendance statistique.**

## 1. Installation dans le checkout isolé ; tests obligatoires

\`\`\`powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch --detach origin/feat/cpu-purged-walkforward-folds
cd backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_h1_short_prospective.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Ne pas lancer si tests rouges.** Les tests sont entièrement synthétiques et n'utilisent aucun service réseau.

## 2. Figer les 3 modèles et verrouiller les 72 heures

Créer le répertoire une seule fois, dans \`E:\cryptoML-walkforward-results\` :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_short_prospective freeze \`
  --data-root "E:\cryptoML\data" \`
  --fold-plan "E:\cryptoML-walkforward-results\p2_feature_reduction_individual_remaining25_full_v1.json" \`
  --output-dir "E:\cryptoML-walkforward-results\p2_h1_short72_v1" \`
  --verify-hashes
\`\`\`

Une fois \`freeze\` terminé, \`protocol.json\` fixe les dates **UTC**, les variables, les empreintes SHA et les itérations retenues. **Ne pas changer la fenêtre en fonction des résultats** ni réentraîner pendant le test.

## 3. Lancer l'observation au premier plan

**Laisser cette fenêtre PowerShell ouverte** (ce n'est pas un service ni un job de fond) :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_short_prospective observe \`
  --experiment-dir "E:\cryptoML-walkforward-results\p2_h1_short72_v1" \`
  --follow
\`\`\`

Le programme attend les nouvelles clôtures, ignore le warmup, tente au plus une prédiction par clôture et s'arrête de lui-même au terme des 72 heures. Un décalage réseau ou un réveil tardif cause une ligne manquante (jamais inventée). \`Ctrl+C\` arrête également l'observation, mais **rend alors le test incomplet**.

## 4. Bilan après arrêt et maturation

Après la fin des 72 heures et au moins 6 minutes de plus :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -u -m mlpipeline.h1_short_prospective evaluate \`
  --experiment-dir "E:\cryptoML-walkforward-results\p2_h1_short72_v1"
\`\`\`

Consulter la comparaison :

\`\`\`powershell
$r = Get-Content "E:\cryptoML-walkforward-results\p2_h1_short72_v1\final_report.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$r.pooled.accuracy | Format-List
$r.paired | Format-List
$r.by_symbol | Format-List
\`\`\`

Même si une précision apparente dépasse 52 %, cela n'établit **aucune** exploitabilité en marché réel. À la fin, arrêt ; **pas de prolongation automatique ni d'ouverture des OOS réservés**.
