# 13 — Journal des décisions (ADR simplifié)

## Confirmées au 12 octobre 2026

| ID | Décision | Motif |
|---|---|---|
| ADR-001 | Next.js + shadcn/ui | Interface choisie par le projet |
| ADR-002 | FastAPI pour backend Python | Même langage que ML, contrats typés |
| ADR-003 | LightGBM, pas Qwen en V1 | Baseline plus directe sur données numériques |
| ADR-004 | Marché perpétuel USD-M | Long/short envisageables lors d'une future phase de stratégie |
| ADR-005 | M5 ; horizons H3/H6/H12 | Comparaison de 15/30/60 minutes |
| ADR-006 | Modèle global sur plusieurs cryptos | Tester apprentissage transférable sur paires inédites |
| ADR-007 | Phase directionnelle avant profits | Éviter de confondre prévision et rentabilité |
| ADR-008 | UP / NEUTRAL / DOWN | Ne pas forcer un signal sur micro-mouvements |
| ADR-009 | Page Qualité prédictive avec replay | Examiner prédiction et vérité sans trades |
| ADR-010 | Lightweight Charts + ECharts | Temps/prix vs statistiques |
| ADR-011 | Walk-forward + holdout symbole | Vérifier généralisation temporelle et cross-asset |

## Choix proposés à confirmer par l'implémentation

| Sujet | Proposition initiale | Méthode de décision |
|---|---|---|
| Historique | ~24 mois si couverture disponible | Vérifier disponibilité, diversité de régimes, stockage |
| Fenêtres walk-forward | Train 120j, validation 14j, test 14j | Tester sensibilité sans regarder le test final |
| Seuil neutre | Volatilité passée × k × sqrt(H) | Comparer labels en validation, figer ensuite |
| Univers d'actifs | 5 paires pilote, 20–30 ensuite | Couverture/liquidité et biais de survivance |
| Feature set | 15–25 features au départ | Ablations et stabilité OOS |
| Stockage SQL | PostgreSQL pour jobs/runs/predictions | Volume d'accès / déploiement réel |
| Queue | worker simple puis RQ/Celery | Besoin de parallélisme/résilience |
| Auth | locale au MVP, sécurisée avant exposition internet | Modèle de déploiement |
| Inférence | à la clôture M5 confirmée | Latence et qualité de données observées |

## Évolutions à documenter plus tard

- Support d'autres exchanges / marges COIN-M.
- Modes baseline binaire vs ternaire.
- Modèles spécialisés par paire ou ensembles global+local.
- Simulation de stratégie avec frais/funding/risque, indépendant du moteur de qualité prédictive.

Toute évolution substantielle doit recevoir un nouvel ADR avec hypothèse, preuve (rapport de validation) et impacts API/UI.
