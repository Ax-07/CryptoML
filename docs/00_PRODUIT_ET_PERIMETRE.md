# 00 — Vision produit, objectifs et limites

## Pourquoi CryptoML ?

Construire une plateforme de **recherche prédictive crypto multi-actifs** qui permet d'apprendre des motifs temporels sur des contrats perpétuels M5, puis de les confronter aux prix réellement observés. L'utilisateur doit pouvoir répondre à :

1. Le modèle anticipe-t-il mieux la direction que des baselines simples ?
2. À quel horizon (15/30/60 min) et sur quels actifs fonctionne-t-il ?
3. Ses probabilités sont-elles fiables et stables dans le temps ?
4. Quels motifs influencent les prédictions et dans quelles conditions échoue-t-il ?
5. Généralise-t-il à une crypto totalement absente des données de développement ?

## Périmètre MVP — Phase prédictive

- Marché des **Futures perpétuels USDT-M** ; premier fournisseur de données : Binance USDⓈ-M.
- Unité de temps : **M5**. 3 horizons distincts H=3, H=6, H=12 bougies.
- Prédiction multiclasses : **UP**, **NEUTRAL**, **DOWN** ; sorties probabilistes.
- Historique multi-crypto : pilote 5 paires, extension ciblée à 20–30 paires liquides.
- Un modèle global sans ID du symbole comme feature au départ ; contexte marché BTC/ETH calculé au même temps.
- Évaluations walk-forward, test final temporel et test final de paires inédites ; visualisation sur chandeliers par replay.
- API interne pour intégrer les évaluations et prédictions à l'application Next.js.

## Hors périmètre MVP

- Passage d'ordres, API privées d'exchange, margin/leverage réels, liquidation, sizing et exécution.
- PnL et backtest de stratégie (pages conservées comme **futures phases**, elles doivent afficher « Non disponible » plutôt que de simuler des résultats réels).
- LLM/Qwen, news et sentiment, modèles deep learning.
- Garantie de prédiction valable sur « toutes » les cryptos, y compris les paires illiquides ou récemment cotées.

## Profils utilisateurs

- **Chercheur/constructeur** : configure et compare données, features, labels, modèles et expériences.
- **Analyste** : examine les matrices de confusion, l'incertitude, le replay et les conditions d'erreur.

## Parcours métier cible

1. Choisir un univers et un intervalle historique ; importer et valider les données.
2. Créer un dataset versionné ; calculer les features/labels sans fuite.
3. Définir horizons et protocole de validation ; lancer un run.
4. Consulter rapports et scores réellement hors échantillon.
5. Cliquer une prédiction ; rejouer le graphique **sans afficher le futur** jusqu'à la maturation du label.
6. Comparer modèles par période, par actif et par classe ; ne promouvoir qu'une version contrôlée.

## Critères de succès — MVP (fonctionnels, pas financiers)

- Sur un jeu de test déterministe, les splits, features, labels et états du replay sont exacts.
- Import des 5 paires pilote avec rapport de qualité et couverture temporelle par symbole.
- Un modèle global H=6 produit des probabilités valides sommant à 1 ; les performances OOS sont mesurées et comparées aux baselines.
- Un résultat de prédiction contient les versions, horodatages, classe attendue et disponibilité du verdict.
- Le replay et la page « Qualité prédictive » utilisent les mêmes identifiants et timestamps que l'évaluation stockée.

## Termes d'interface

- **Prediction/Prévision** = distribution P(UP, NEUTRAL, DOWN) sur un horizon.
- **Classe réelle** = catégorie calculée après disponibilité de la dernière bougie de l'horizon.
- **Signal de trading** = *non fourni dans le MVP*, distinct d'une prévision directionnelle.
- **Backtest de trading** = future phase ; ne pas confondre avec le backtest prédictif hors échantillon.
