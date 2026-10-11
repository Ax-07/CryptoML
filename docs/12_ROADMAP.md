# 12 — Roadmap de réalisation

## P0 — Fondations (faire avant le design d'entraînement avancé)

**Livrables :** backend Python initial, schéma du marché, ingestion Futures Binance 5m, contrôle qualité, stockage Parquet.

- [ ] Inspecter projet Next.js déjà créé et définir le monorepo sans écraser de fichiers.
- [ ] Configurer venv, formatage, tests, CI simple ; FastAPI `/health`.
- [ ] Télécharger cinq paires pilote et construire un manifest de dataset versionné.
- [ ] Détecter trous et anomalies, synchroniser contexte BTC/ETH.
- [ ] Exporter `quality_report.json` et un petit fixture de test.

## P1 — Premier résultat prédictif solide

**Livrables :** features V1, labels H6, modèle de référence, LightGBM global, métriques OOS.

- [ ] Définir précisément `bar_close_at`, `label_mature_at`, `available_at`.
- [ ] Implémenter 15–25 features sans fuite et labels `DOWN/NEUTRAL/UP`.
- [ ] Écrire tests point-in-time et anti-fuite.
- [ ] Mettre en place splits temporels purgés / baseline / LightGBM H6.
- [ ] Enregistrer run, modèles et prédictions OOS.
- [ ] Générer confusion, macro F1, balanced accuracy, log loss et calibration.

## P2 — API et vrai replay

**Livrables :** endpoints versionnés, historique et page Qualité prédictive connectée.

- [ ] Activer PostgreSQL si besoin de stockage runs/jobs partagé.
- [ ] Implémenter endpoints `candles`, `models`, `evaluations`, `predictions`, `replay`.
- [ ] Intégrer Lightweight Charts / contrôles Play-Pause-Step.
- [ ] Afficher probabilités, marqueurs, échéance de vérité et erreurs réelles.
- [ ] Tester la confidentialité temporelle côté serveur.

## P3 — Multi-crypto robuste et plusieurs horizons

**Livrables :** 20–30 paires, H3/H6/H12, walk-forward sur période et paire inconnue.

- [ ] Ajouter paires avec filtre de qualité/liquidité et historique de listing.
- [ ] Évaluer balances de classes et pondération par symbole.
- [ ] Comparer horizons, modèle sans symbole vs ablations, contexte BTC/ETH.
- [ ] Mettre en place test final temporel et test final symboles inédites.
- [ ] Ajouter SHAP, suivi d'expériences, comparaison de modèles dans l'UI.

## P4 — Inférence fraîche et fiabilisation

**Livrables :** prédictions post-clôture M5, résolution différée de labels, monitoring.

- [ ] Orchestrer import/feature/inférence à chaque nouvelle clôture valide.
- [ ] Stocker les versions et traçabilité de chaque événement.
- [ ] Ajouter vérification fraîcheur, dérive de features et alerte sur pipeline défaillant.
- [ ] Versionner promotion de modèles et revue humaine des résultats.

## P5 — Plus tard : recherche de stratégies

Conversion facultative des prévisions en stratégies LONG/SHORT/WAIT, frais Futures, funding, spread, slippage, marges/liquidations, exposition cross-asset ; **pas d'ordres réels par défaut**. Concevoir un cahier des charges de risque distinct avant toute intégration de trading.
