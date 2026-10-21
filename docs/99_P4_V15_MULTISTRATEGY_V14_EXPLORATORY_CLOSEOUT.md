# P4 V15 — Clôture exploratoire V14 : cinq stratégies et robustesse A01

**Date 21 octobre 2026. Statut : conclusions fondées uniquement sur les sorties PowerShell transmises par l'opérateur.** Aucune inspection des CSV bruts depuis GitHub (fichiers locaux à l'opérateur), aucune vérification indépendante des agrégats hors tests déjà rapportés, aucun changement de signal, aucun entraînement ML ni ordre.

## Intégrité de l'exécution

- V13 : 6 211 signaux (sur quatre actifs), témoins de provenance vérifiés par V14 avec message `V13 FINGERPRINT + EVENTS MATCH`.
- V14 : 6 206 opportunités évaluées, 5 écarts entre signaux recensés et opportunités maturées ; détail des causes d'exclusion à extraire du JSON local avant clôture exhaustive.
- L'opérateur a confirmé `pytest -q` complet à 100 % sans échec, avertissements Starlette / LightGBM seulement.

## Scores V14 : moyenne par signal hypothétique, hypothèses de frais aller-retour

| Stratégie | n | brut bp | net 2 bp | net 5 bp | net 10 bp |
|---|---:|---:|---:|---:|---:|
| A01 MOMENTUM H4 | 822 | +5,0053 | +3,0053 | +0,0053 | −4,9947 |
| A02 DONCHIAN H1 | 2 266 | −4,3967 | −6,3967 | −9,3967 | −14,3967 |
| A03 COMPRESSION H1 | 675 | −10,5865 | −12,5865 | −15,5865 | −20,5865 |
| A04 MEAN REVERSION H1 | 409 | −1,8734 | −3,8734 | −6,8734 | −11,8734 |
| A05 V9 RETEST | 2 034 | −1,4111 | −3,4111 | −6,4111 | −11,4111 |

Ces mesures ne tiennent compte ni du funding, ni des coûts d'exécution réels, ni de portefeuille. Elles sont produites par stop ATR1/target ATR1,5/timeout propres à chaque famille, et sont des **markouts théoriques qui peuvent se recouvrir**, non PnL d'un capital donné.

## A01 Momentum H4 : constat d'instabilité

**Par actif et direction (net 5 bp moyen)**

| Actif | LONG (n / bp) | SHORT (n / bp) |
|---|---|---|
| BTCUSDT | 102 / −0,27 | 98 / −22,40 |
| ETHUSDT | 114 / +26,66 | 103 / −0,64 |
| SOLUSDT | 99 / +6,83 | 112 / +9,22 |
| XRPUSDT | 93 / +19,63 | 101 / −42,37 |

**Dans le temps** : 11 mois positifs sur 21 après 5 bp. Janvier–décembre 2025 par exemple comprennent février +152,59 bp, mars −132,57 bp, septembre +17,49 bp, octobre −90,47 bp, décembre −108,14 bp. Ce sont des moyennes mensuelles par événement, non des performances mensuelles de portefeuille.

**Verdict méthodologique** : A01 ne possède **pas** un avantage robuste démontré après 5 bp ; le rendement moyen +0,0053 bp est économiquement nul. Le brut positif justifie de comprendre davantage le mécanisme, mais **pas** un passage live ou ML. Ne pas filtrer ETH LONG / SOL ou supprimer BTC SHORT / XRP SHORT en regardant cet historique : ce serait une optimisation post hoc.

**Verdict A02–A05** : les configurations préenregistrées sont négatives avant frais ; elles ne démontrent aucun avantage. Cela ne permet pas de condamner les familles entières en général.

## Orientation du prochain cycle, sans backtest additionnel aujourd'hui

1. Compléter le diagnostic **sans sélectionner les meilleurs segments** : exclusions (5), évolution hebdomadaire, distribution (quantiles), drawdown de portefeuille avec calendrier et positions concurrentes (non disponible avec les seules moyennes), durée réelle, prix de sortie en cas de gaps, funding/slippage et examen critique de l'ATR.
2. Écrire **avant résultats** un protocole de nouvelles familles B01 carry delta-neutre / basis et éventuellement A06 confirmation volume-taker si données valides. Éviter simplement de changer les mêmes EMA/Donchian sur un échantillon ayant déjà montré l'absence de robustesse.
3. Toute stratégie nouvelle sera évaluée selon une politique de simulation adaptée et un registre des hypothèses testées, avec comparaisons NO_TRADE, stratégies triviales et portefeuilles à risques comparables.
4. Méta-modèle multi-stratégies uniquement si plusieurs experts possèdent une justification économique : apprentissage causal hors labels futurs, prédictions de gain net conditionnel + **NO_TRADE** ; backtest historique exploratoire puis validation prospective réellement nouvelle.

## Conclusion

**NE PAS déployer V10 live, NE PAS entraîner un sélecteur de « meilleur backtest » sur les 21 mois inspectés.** Les cinq premières stratégies constituent un résultat de recherche négatif ou non concluant, bien documenté. Notre feuille de route reste la recherche de mécanismes économiquement distincts, surtout carry/funding/basis et vérification de la tendance sur données nouvelles.
