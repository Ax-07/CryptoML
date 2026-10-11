# 03 — Features et labels de prévision directionnelle

## Définition temporelle

Pour une paire `s` et une bougie clôturée à l'instant `t` :

- **Features `X(s,t)`** : exclusivement des informations de bougies clôturées au plus tard à `t`, et disponibles lors de l'émission.
- **Close de référence `C_t`** : clôture de la dernière bougie connue.
- **Label H** : prix à la clôture de la **H-ième bougie future** (`C_(t+H)`), avec `H ∈ {3, 6, 12}`.
- **Rendement réalisé** : `r(s,t,H) = C_(t+H) / C_t - 1`.
- `label_mature_at = close_time(t+H)` ; un label n'existe pas avant cet instant.

Ce rendement est une **mesure de marché**, pas un rendement réalisable après exécution. L'entrée sur clôture connue n'est pas utilisée ici comme hypothèse de trading.

## Classes

La cible est `DOWN = 0`, `NEUTRAL = 1`, `UP = 2`. La politique de seuil est versionnée.

Version initiale envisagée : `tau(s,t,H) = k_H * volatility_past(s,t) * sqrt(H)` où `volatility_past` est la déviation standard des rendements M5 sur une fenêtre passée (par exemple 48 bougies), avec garde contre zéros/anomalies. `k_H` est défini sur **train/validation exclusivement**, jamais sur le test final.

- `UP` si `r > +tau` ; `DOWN` si `r < -tau` ; `NEUTRAL` sinon.
- La classe NEUTRAL exprime un mouvement **de faible amplitude**, pas la recommandation `WAIT` d'une stratégie.
- Comparer en ablation : label binaire hausse/baisse, ou seuil fixe en % **fixé sans test leakage**.
- Si distribution des classes très déséquilibrée : vérifier proportion et seuil, ne pas interpréter une accuracy brute élevée comme succès.

**Attention** : une alternative « mouvement significatif touché au cours de H » utilisant high/low et barrières est une autre cible, à spécifier séparément. Le MVP vise uniquement `close à horizon fixe`.

## Features candidates (point de départ, à ablater)

| Groupe | Features | Fenêtres M5 |
|---|---|---|
| Rendements | `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`, `ret_48` | 5 min → 4h |
| Volatilité | `std_ret_12`, `std_ret_48`, `atr_pct_14` | 1h, 4h, 70min |
| Tendance | `close/ema_12-1`, `close/ema_48-1`, `ema_12/ema_48-1` | 1h, 4h |
| Oscillateur | `rsi_14` | 70min |
| Structure | `body_pct`, `range_pct`, `upper_wick_pct`, `lower_wick_pct`, `close_position` | bougie actuelle clôturée |
| Volume | `volume_quote_rel_12`, `volume_quote_rel_48`, `taker_buy_share` | historique exclusivement |
| Contexte | `btc_ret_3`, `btc_ret_12`, `eth_ret_3`, `eth_vol_12` | mêmes horodatages |
| Saisonnalité | `sin/cos(hour_utc)`, `sin/cos(weekday)` | date connue |

Débuter avec **15–25 features validées** puis enrichir. Éviter les prix absolus comme unique signal de l'identité d'une crypto ; privilégier pourcentages et ratios. Des features de liquidité en USDT pourront être ajoutées comme ablation contrôlée.

## Formules exemplaires

```python
# Hypothèse : données triées par (symbol, bar_close_at).
# Transformations par paire, JAMAIS rolling global mélangeant les actifs.
g = df.groupby('symbol', sort=False)
df['ret_6'] = g['close'].pct_change(6)
df['ema_48'] = g['close'].transform(lambda s: s.ewm(span=48, adjust=False).mean())
df['distance_ema_48'] = df['close'] / df['ema_48'] - 1

# Calcul du futur UNIQUEMENT pour créer la cible / l'évaluation :
for h in (3, 6, 12):
    future_close = g['close'].shift(-h)
    df[f'future_return_{h}'] = future_close / df['close'] - 1
    # label_mature_at = g['bar_close_at'].shift(-h)
```

Dans l'implémentation réelle, vérifier qu'aucune discontinuité de temps ne traverse la fenêtre de calcul, même si les lignes suivantes existent. Les features BTC/ETH doivent être jointes par timestamp sans interpolation future.

## Contrat de features

Artefact `feature_schema.json` : noms ordonnés, types, unités, fenêtres, politique NaN, versions des calculs, contexte exigé, timezone et timestamp d'émission. Le même builder et la même version sont utilisés pour train, test et inférence.

## Interdictions et contrôles

- Jamais de `.shift(-n)` dans les features ; réserver le décalage négatif aux labels.
- Aucun scaler/global encoder appris sur des données de validation ou test.
- `future_return`, `actual_class`, `label_mature_at`, informations postérieures et `symbol` identifiant sont interdits dans `X` initial (symbol reste une **clé de jointure**).
- Fenêtre d'indicateurs incomplète → sample marqué `not_eligible`, pas de remplissage par futur.
- Unit tests : modifier tous les futurs après `t` ne doit **jamais** changer `X(s,t)` ni la probabilité prédite pour un modèle figé.
