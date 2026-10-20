# CryptoML V10 — Observateur Binance Futures public, suivi papier local

**STATUT : code publié, tests locaux à effectuer. Aucune rentabilité établie, aucun ordre réel, aucune clé API.** V9 reste une hypothèse de recherche. Cette V10 n'utilise que l'endpoint REST public Binance USD-M \`GET /fapi/v1/klines\`, agrège les bougies clôturées et déclenche le détecteur V9. Elle n'utilise pas \`POST\`, n'envoie aucun ordre et n'accède pas au compte.

## Protocole de preuve : limitations honnêtes

- L'application interroge les bougies M5 des quatre cryptos BTC/ETH/SOL/XRP et refuse d'utiliser la bougie actuellement en formation. Les 600 bougies les plus récentes servent à initialiser M15/H1 et la structure.
- Elle rejette des bougies finales trop anciennes (>10 min), les erreurs de données et toute décision de retest qui ne correspond **pas** à la dernière clôture du flux. Elle ne reconstitue pas silencieusement des signaux tardifs en prétendant les avoir observés en direct.
- Les cassures et décisions doivent être >= 2026-10-22 00:00 UTC ; les classes H1 et niveaux M15 sont déterminés selon V9. Les signaux sont enregistrés en SQLite avec clé unique ; relancer le même scan ne duplique pas une observation.
- Chaque ligne consigne le moment d'observation de la machine, les horodatages de cassure/retest, le SHA256 des bougies fournies au scanner, le protocole et le contexte H1.
- **Un SQLite local n'est pas une preuve infalsifiable ni un enregistrement externe horodaté** ; l'horloge PC peut dériver, la capture peut être retardée, le fichier peut être modifié. Statut explicite \`LOCAL_OBSERVATION_NOT_EXTERNAL_ATTESTATION\`. Une validation véritablement indépendante nécessite une source de confiance, une gestion de l'intégrité du journal hors processus et un protocole de fills/coûts fixé à l'avance.
- Ne pas prendre de décisions financières ou d'ordres à partir de ce scanner. Aucun rendement n'est calculé.

## Windows PowerShell

\`\`\`powershell
cd E:\cryptoML-structure-v9
git fetch origin
git worktree add E:\cryptoML-paper-observer-v10 origin/research/v9-binance-paper-journal-v10
cd E:\cryptoML-paper-observer-v10\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_binance_paper_observer_v10.py tests/test_structure_breakout_retest_v9.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
\`\`\`

**Si les tests passent**, exécuter une fois le scanner et créer le journal :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.binance_paper_observer_v10 \`
  --journal "E:\cryptoML-walkforward-results\p4_v10_paper_observations.sqlite3"
\`\`\`

Les exécutions futures réutilisent le même fichier SQLite : les entrées existantes sont conservées. Le fichier est **hors P0**. Aucun accès automatique n'est planifié par le logiciel livré : la commande est un scan ponctuel. Pour un suivi à chaque clôture M5, une programmation locale fiable est requise avant d'interpréter les absences de signaux. L'accès réseau public Binance doit être autorisé et fonctionnel ; une erreur HTTP/régionale n'est pas une absence de signaux.

Affichage local, **lecture seule** :

\`\`\`powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -c "import sqlite3; c=sqlite3.connect(r'E:\cryptoML-walkforward-results\p4_v10_paper_observations.sqlite3'); print(*c.execute('select symbol,side,decision_close_at,first_observed_utc,provenance from observed_signals order by first_observed_utc desc limit 20').fetchall(),sep='\n')"
\`\`\`

## Précautions

- Binance peut limiter ou restreindre ses API suivant la zone et ses conditions d'utilisation ; ne pas contourner un blocage, et ne pas substituer des données d'un autre marché en cachant la provenance.
- Éviter de lancer plusieurs processus concurrents sur le même journal.
- Une commande ponctuelle génère souvent **zéro** nouvelle observation ; cela ne signifie pas que le marché n'offre jamais de cassure/retest.
- Le programme n'invente ni stop, ni objectif, ni levier, ni slippage/funding. Une autre étape figée devra définir les issues économiques avant d'observer le futur.
