# 11 — Stratégie de tests et critères d'acceptation

## Tests unitaires data

- Normalisation des timestamps Binance : intervalle M5, `bar_close_at` exclusif, conversion millisecondes/secondes.
- Dédoublonnage archive + REST idempotent.
- Détection de trous de 5 min, d'OHLC invalides et de bougies non clôturées.
- Rejet d'une fenêtre de features/label traversant un trou.
- Séparation des calculs par `symbol` : une variation BTC ne doit pas altérer un indicateur historique SOL hors features contextuelles prévues.

## Tests anti-fuite (bloquants)

1. Construire `X(s,t)` sur un jeu connu. Modifier toutes les bougies `>t` : **X(s,t) reste identique**.
2. La feature `close/ema` n'utilise pas une moyenne centrée ni des futures ; `taker_buy` de la bougie en cours uniquement après clôture.
3. Aucun `label`, `future_return` ni timestamp futur n'est présent dans `X`.
4. Le builder multi-timeframe ne joint jamais une bougie H1 partiellement formée.
5. Purge : tous les samples train satisfont `label_mature_at < validation_start` (selon le contrat de borne retenu) et de même pour validation/test.
6. Test final et paires inédites n'influencent jamais selection d'hyperparamètres/seuil.
7. L'égalité des schémas de features entre train et inférence est vérifiée au chargement modèle.

## Tests évaluation

- Probabilités finies, dans `[0,1]` et somme de 1 à tolérance flottante.
- Classes correctement encodées `DOWN=0, NEUTRAL=1, UP=2`.
- Confusion matrix avec lignes vrai, colonnes prédit ; effectifs et supports cohérents.
- Macro F1, balanced accuracy, log loss et Brier calculés vs un fixture connu.
- Scores rapportés par horizon, fold, crypto, segment OOS et avec nombre d'observations.
- Aucune période OOS dupliquée dans un agrégat final.
- Cas monoclasse détecté et métrique impossible indiquée « indisponible » plutôt que faux 0/100 %.

## Tests API

- Pydantic rejette symboles inconnus, horizons invalides, `from > to`, très grandes plages sans pagination.
- `POST training-runs` retourne 202 et un ID ; transition job cohérente, statut erreur explicite.
- API retourne UTC ISO Z ; adaptateur chart gère timestamps en secondes.
- Contrat OpenAPI validé en CI, tests de contrat contre les exemples JSON.
- Cache de données et droits n'exposent pas d'évaluation final-test en mode développement si gel requis.

## Tests replay (essentiels)

- Curseur antérieur à `prediction_issued_at` → **prévision absente**.
- Curseur antérieur à `label_mature_at` → `actual_class=null`, `actual_return=null`, jamais présent dans le payload.
- Curseur à maturation et donnée disponible → verdict révélé.
- Reculer avant maturation → verdict redevient masqué.
- L'API n'envoie aucune bougie avec `bar_close_at > cursor_at`.
- Sélection d'une prédiction n'autorise pas l'affichage automatique d'un futur caché.
- Une prédiction générée in-sample ne peut pas apparaître dans le segment OOS.

## Tests interface

- Navigation entre **11 pages** sans lien mort ; pages futures signalées comme telles.
- Filtre `symbol / horizon / model_version / evaluation` appliqué partout, y compris charts et panels.
- Écran vide, erreur API, chargement, dataset incomplet et modèle absent correctement rendus.
- Replay accessible au clavier, mobile utilisable, pastille « demo » sur séries de démonstration.

## Definition of Done — Première milestone

- Données pilote versionnées et validées ; tests anti-fuite verts.
- Training H6 reproduisible ; baseline + LightGBM avec métriques OOS réalistes, même si le gain est nul.
- API retourne des prédictions historiques avec vérité différée.
- Page « Qualité prédictive » connectée aux vraies réponses et replay conforme.
- README + docs mis à jour et instructions de lancement réellement exécutées.
