# CryptoML — P4 Trend Pullback ML V1 (LONG, M5)

**Statut : spécification + détecteur causal + étiquetage indicatif + tests synthétiques. Pas d'entraînement ni de résultats historiques.**

## Hypothèse
Une IA peut-elle sélectionner, parmi les premiers pullbacks vers l'EMA9 d'une tendance haussière EMA9 > EMA26, des entrées LONG dont le résultat futur est meilleur que (A) prendre tous les événements et (B) attendre une confirmation mécanique ? Une amélioration historique n'est ni une preuve de rentabilité ni une validation indépendante.

Branche indépendante `research/trend-pullback-ml-v1` basée sur `feat/cpu-purged-walkforward-folds`. P0, P2, P3, ADA, anciens checks, OOS et holdout final déjà consulté ne sont pas modifiés. Les données historiques réutilisées ne pourront pas être présentées comme un nouveau test indépendant.

## V1 — Définition figée de l'événement

Univers de recherche : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT ; chandeliers Binance USD-M 5 minutes **clôturés**, timestamps UTC. Un symbole à la fois. Les trous M5 réinitialisent EMA et ATR.

EMA9/EMA26 : moyenne exponentielle `adjust=False`, min_periods=9/26. ATR14 : moyenne arithmétique glissante 14 de true range ; première valeur de TR de chaque segment indisponible. À t-1 : `close > EMA9 > EMA26`, deux EMA en hausse comparées à t-4, `low(t-1) > EMA9(t-1)`. À t : `low(t) <= EMA9(t)` (premier contact), `EMA26(t) <= close(t) <= close(t-1)`, `low(t) >= EMA26(t) - 0.25 ATR14(t)`, ATR strictement positif. La décision est connue **après close(t)**, jamais au contact intrabar. Minimum 8 bougies entre événements d'un même segment. Un signal brut rejeté pour cooldown n'est pas transformé en trade.

Le détecteur `backend/mlpipeline/trend_pullback.py` retourne indicateurs, état tendance, premier contact et signal retenu ; **aucun label ni prix futur**.

## Baseline A — Résultat indicatif, pas un backtest exécutable

Entrée LONG hypothétique à l'OPEN de la bougie qui commence **5 minutes après la clôture du signal** (on laisse passer une bougie complète). Stop = prix d'entrée moins `1 x ATR14(t)`. Objectif = entrée plus `1.5 x ATR14(t)`. Observer exactement 6 bougies à partir de l'entrée, soit jusqu'à **t+35 min**. Premier stop/objectif touché gagne ; si les deux sont touchés pendant la même bougie, compter le stop. Si ouverture au-delà du stop : sortie au pire prix d'ouverture. Ouverture favorable au-delà de l'objectif : sortie plafonnée à l'objectif. Sans déclenchement : sortie au CLOSE de la sixième bougie.

Maturité conservatrice à t+35 min même si barrière touchée plus tôt. Exclure bougies manquantes et événements non mûrs au cutoff autorisé. `gross_bps = 10000 * (sortie/entrée - 1)` ; sensibilités indicatives de coût aller-retour total 2/5/10 bp. L'entrée au prix OPEN futur n'est pas garantie. Ne pas additionner ces markouts comme un PnL de portefeuille. Le calcul des issues est **strictement séparé** de la construction des features : `backend/mlpipeline/pullback_outcomes.py`.

## Baseline B et candidat C — À faire dans une prochaine étape

B : règle mécanique de confirmation à définir **avant** examen des résultats, avec prix d'entrée postérieur à la confirmation. Ne jamais créditer rétroactivement l'entrée anticipée de A.

C : classifieur simple régularisé puis, si nécessaire, LightGBM, sur les seuls événements retenus et features connues à close(t) ; cible = issue de A à horizons/coûts identiques. Toujours autoriser l'abstention. Comparer A/B/C sur les **mêmes opportunités** en indiquant séparément couverture et résultat par opportunité/trade. Aucune sélection après examen des données réservées.

## Protocole de validation

1. Tests synthétiques offline : modifier tout l'avenir après t sans changer les décisions passées ; rejeter les bougies ouvertes/invalides ; reset après trous ; pas de résultat immature ; entrée différée et stop prioritaire en cas d'ambiguïté intrabar.
2. Examiner la fréquence et la géométrie des événements uniquement sur une période historique **explicitement autorisée pour le développement** ; exclure pour le moment toutes données après `2026-07-01T00:00:00Z` jusqu'à audit de provenance. Les anciennes fenêtres explorées ne sont pas indépendantes.
3. Geler schéma de features, labels et seuils avant tout entraînement ; train/early/tuning chronologiques, purge/embargo d'au moins 35 minutes à chaque frontière ; pas de données d'autres paires futures dans les features.
4. Les comparaisons historiques sont exploratoires. Toute conclusion confirmatoire exige une collecte réellement prospective de prédictions scellées avant maturité, horodatées de manière externe, coûts d'exécution réalistes et analyses par jour et par actif.
5. Aucun ordre, effet de levier, automatisation trading ou déploiement réel dans cette V1.

## Tests sans accès aux datasets

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
# Utiliser un NOUVEAU worktree, ne pas basculer le checkout historique
git worktree add E:\cryptoML-trend-pullback origin/research/trend-pullback-ml-v1
cd E:\cryptoML-trend-pullback\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_trend_pullback.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Pour effectuer des commits locaux, depuis ce nouveau worktree en HEAD détachée : `git switch -c research/trend-pullback-ml-v1 --track origin/research/trend-pullback-ml-v1`.

**Ces scripts n'exécutent aucun entraînement et n'ouvrent aucune période réservée.**
