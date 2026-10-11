# 08 — Replay « Prédit vs Réel » : spécification fonctionnelle

## Finalité

Examiner la qualité d'anticipation **sans trade**. Rejouer les bougies M5 d'un actif, afficher uniquement les prévisions qu'un modèle aurait pu émettre, puis révéler le résultat réel à maturité de l'horizon.

## Éléments du panneau

1. **Chart principal** : chandeliers M5 du perp choisi ; marqueur `UP` vert, `DOWN` rouge, `NEUTRAL` gris à la bougie de référence.
2. **Chart probabilités** : séries P(UP), P(NEUTRAL), P(DOWN) alignées sur timestamps de prédiction, sans point futur.
3. **Contrôles replay** : Play/Pause, pas de +1/-1 bougie, vitesse, retour, curseur temporel, saut à prédiction, reset.
4. **Inspecteur** : heure d'émission, modèle/fold, H, probabilités, classe prédite, classe réelle **si disponible**, rendement réel **si mature**, résultat correct/incorrect et facteurs explicatifs.
5. **Liste/timeline des prévisions** : événements émis à une date <= `cursor_at` ; un clic sélectionne l'événement en conservant les restrictions temporelles.
6. **Options** : chandeliers/courbe, affichage des marqueurs, segment OOS, comparaison multi-horizons facultative (après MVP).

## États temporels

| État | Condition | UI |
|---|---|---|
| `NOT_ISSUED` | `prediction_issued_at > cursor_at` | prévision invisible |
| `PENDING` | prédiction émise, mais `label_mature_at > cursor_at` | probas visibles ; verdict masqué |
| `MATURED` | `label_mature_at <= cursor_at`, données complètes | vraie classe/rendement visibles, succès/erreur |
| `UNAVAILABLE` | horizon manquant/invalidé | absence de verdict avec explication |

**Règle absolue :** l'API replay n'envoie aucune bougie dont la clôture est postérieure au curseur. Le frontend applique aussi le filtrage pour l'UX, mais le backend doit le garantir. Un modèle prédit à l'instant `t` n'a pas accès aux bougies `t+1...`.

## Cas particulier : prévisions OOS historiques

Dans un backtest prédictif, le modèle d'un fold est entraîné **avant** le début du segment OOS. Chaque prévision doit pointer vers `evaluation_id`, `fold_id` et `model_version` qui existaient à cette date simulée. Ne jamais calculer des prévisions « historiques » avec un modèle entraîné sur ces mêmes observations puis les présenter comme OOS.

## Chronologie exemple

```text
12:00 — bougie de 11:55–12:00 clôturée, disponible après latence.
12:00:02 — modèle H6 émet P(UP)=0.62, P(NEUTRAL)=0.21, P(DOWN)=0.17.
12:05 — replay avance : nouvelle bougie visible, verdict H6 reste PENDING.
12:25 — dernière bougie visible à 12:25, verdict toujours PENDING.
12:30 — clôture H6 disponible : comparer close(12:30) / close(12:00)-1.
12:30:02 — verdict mature et devient consultable (selon latence de données).
```

Cette convention suppose qu'une bougie `12:00` est nommée par **son instant de fin**, la convention `bar_close_at` décrite dans le contrat.

## Critères d'acceptation

- Pause à 12:10 : aucune bougie close après 12:10, aucun verdict d'une prévision 12:00/H6.
- Déplacement vers une prédiction future : elle reste masquée si son émission dépasse `cursor_at`, sauf choix explicite de déplacement du curseur.
- Reculer le replay **masque à nouveau** les verdicts devenus immatures.
- Passer H3→H6 met à jour toutes les prévisions/verdicts selon le nouvel horizon ; conserver l'indicateur de date et le symbole.
- Cliquer une prédiction erronée l'affiche avec sa situation d'époque et ses features de l'époque, sans explication fabriquée après coup.
- Sur mobile, contrôles de lecture accessibles et graphiques synchronisés sans fuite de données.

## Implémentation technique

Un `ReplayState` minimal : `{evaluation_id, symbol, horizon_bars, cursor_at, playing, speed, selected_prediction_id}`. Une requête `/api/v1/replay` retourne uniquement les objets autorisés au curseur. L'interface affiche `DEMO` si les séries sont synthétiques et `OOS` si elles proviennent d'une évaluation hors échantillon.
