# P2 — Protocole préspecifié de sélection **interne** H2, sans accès aux checks

**Objet :** remplacer les comparaisons opportunistes sur les 36 checks déjà inspectés par une procédure isolée **train / early / tuning**. Les décisions futures ne seront PAS fondées sur les anciens checks. Aucun ADA, OOS de fold, dernier holdout 90 j ni P3.

## Candidats et prédiction (inchangés)

Deux architectures **figées** à l'avance, copiées strictement de `backend/mlpipeline/h2_delayed_selective.py` :
- `regression_30` : LightGBM regressor CPU 30 features, 150 itérations max, `learning_rate=.04`, `num_leaves=31`, `min_child_samples=100`, `reg_lambda=5`, `max_bin=63`, seed 42, early stopping 30 sur la partition early seulement.
- `ridge_ema2` : `StandardScaler + Ridge(alpha=10)`, EMA2, train seulement.
- **Référence abstention totale** : net 0, sans modèle appris.

Rendement ciblé en bp = `10000 * (close(t+10min)/open(t+5min) - 1)` : signal après clôture t, attente d'une bougie entière, entrée hypothétique à l'open de la bougie de t+5, sortie hypothétique au close t+10 ; prix **non garantis**, aucun fill ou PnL exécutable. **LONG** strictement si prévision > +3 bp, **SHORT** si < −3 bp, sinon ABSTENTION. Coût fixe primaire **2 bp aller-retour par transaction** ; sensibilités descriptives 0/5 bp, **sans** optimiser le seuil.

## Construction causale autorisée

36 folds P2 originaux (train 120 j, early 7 j, tuning 3 j), H12-purge 60 min à chaque séparation. Charger UNIQUEMENT BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT depuis P0 en lecture seule (24 mois, fingerprint identique au plan); clip par fold **à `tuning_end = check_start`, avant la construction de toute feature/étiquette prospective du fold**. L'export **ne produit aucune prédiction et ne calcule aucune métrique** pour les checks précédemment inspectés. Aucune étape ne parcourt les checks OOS ou périodes réservées. Comparer les tailles train, early et tuning au rapport original; arrêt strict si décalage de géométrie, label immature, gap ou empreinte.

Les modèles sont entraînés sur les exemples train, arrêt anticipé sur early (seul LightGBM), **prédictions uniquement sur tuning**. Toutes les métriques descriptives proviennent exclusivement des anciens 3 jours de tuning (déjà utilisés historiquement pour des critères de couverture : **pas de validation indépendante**, ni licence de promouvoir des modèles). Aucun réentraînement train+early+tuning pendant cette expérience.

## Critère de promotion prudent et immuable

Calculer pour chaque architecture en 36 folds : rendement net à 2 bp **par opportunité** (abstentions incluses), par trade, nombre et couverture de trades, signes des contributions par fold, métriques par symbole et journée UTC, amplitude et concentration des meilleurs trades, et test descriptif de retrait de chaque fold (sans réentraînement). Les folds sont chevauchants à l'entraînement et ne sont pas indépendants.

Un candidat est **éligible** seulement si TOUTES les conditions figées sont remplies :
1. au moins **200 trades** sur tous les tuning, une couverture totale au plus **5 %**;
2. moyenne nette à 2 bp par trade ET par opportunité **strictement positives**;
3. au moins **12 folds actifs** et au moins **60 %** des folds actifs avec somme nette **strictement positive**;
4. au moins **3/4 actifs** ont un résultat net par trade positif (et au moins 20 trades chacun);
5. résultat net des trades **strictement positif après retrait de chaque fold** et **après suppression descriptive des 5 meilleures transactions théoriques**.

En cas de plusieurs éligibles : choisir **la régression Ridge EMA2** si éligible (simplicité prédéfinie), sinon LightGBM 30. Sinon statut `NO_ELIGIBLE_CANDIDATE`. **Ce choix est une décision de développement interne, pas une validation statistique**, sans revue des checks. Ne pas optimiser ces seuils après résultats. Préconiser une nouvelle évaluation indépendante uniquement quand l'architecture complète sera figée, sans ouvrir les périodes réservées à cette étape.

## Sorties

Un JSON complet (protocole, fingerprint, paramètres, agrégats, éligibilité, choix ou abstention de choix, avertissements), un CSV comparatif et un Parquet des **seules prédictions tuning** sous le répertoire de résultats de recherche isolé. Les anciens exports des checks ne sont **pas des entrées**. Aucun ordre, aucune surveillance P3 et aucun service activé.
