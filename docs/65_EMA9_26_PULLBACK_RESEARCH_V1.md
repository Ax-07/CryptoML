# Recherche V1 — Tendances EMA9/EMA26 et pullbacks haussiers

**Statut : protocole exploratoire et tests synthétiques ; aucun modèle promu, aucun ordre.**
**Branche :** `research/ema9-26-pullback-v1`, issue de `feat/cpu-purged-walkforward-folds`.
Ce travail n'utilise ni les labels ADA, ni les tests OOS/final90 réservés, ni les journaux P3.

## Question

Dans une tendance haussière définie *à la clôture de la bougie M5* par
`close > EMA9 > EMA26`, peut-on détecter un repli vers EMA9 suivi d'une
reprise qui prédit la couleur de la **prochaine bougie clôturée** mieux
qu'un contexte de tendance seul ? Ensuite, le signal possède-t-il un
markout retardé positif **après coûts** ? Ces deux questions sont distinctes.

**Attention :** les EMA alignées ne prouvent pas à elles seules une tendance
persistante. Les règles ci-dessous sont une hypothèse figée *avant* le
premier résultat de cette nouvelle étude, et non une vérité de marché.

## Définition causale initiale, LONG uniquement

Tout est calculé sur des bougies M5 *clôturées* ; les trous remettent
EMA/ATR/rolling à zéro. Au signal t :
1. `close[t] > EMA9[t] > EMA26[t]` et `EMA9[t] > EMA9[t-3]`.
2. À t-3, `close > EMA9 > EMA26` (tendance préexistante).
3. Parmi t-3, t-2, t-1, au moins une bougie a `low <= EMA9 + 0.25 * ATR14`
   et `low >= EMA26 - 0.25 * ATR14`, avec EMA9 > EMA26.
4. La bougie de reprise t est verte (`close > open`) et sa clôture dépasse
   la clôture précédente.
5. Au plus un signal tous les 8 intervalles M5, par segment/symbole.

Les constantes `0.25 ATR`, 3 bougies et 8 bougies sont **des paramètres V1
fixes, pas optimisés**. Aucun SHORT V1, aucune sélection a posteriori par
crypto ou heure. Une version vendeuse sera un protocole séparé.

## Deux cibles différentes : direction et exécution théorique

**Prédiction prochaine bougie :** `next_bar_up = 1` si
`close(t+5min) > close(t)`, sinon 0. Elle ne présuppose aucun fill.

**Markouts de recherche :** signal disponible après close(t) ;
on laisse passer une bougie complète puis on suppose une entrée à
l'OPEN de la bougie commençant t+5 min, et une sortie au CLOSE après
H=1/2/3/6 bougies depuis cette entrée. Coût fictif primaire : 2 bps
aller-retour ; sensibilités descriptives 0 et 5 bps ultérieurement.
La cible est écartée si une bougie manque ou si sa maturité est
`>= end_exclusive`. Aucune garantie d'obtenir l'open, le close ou les
frais supposés. Ce n'est ni un backtest exécutable ni une simulation de capital.

## Expériences dans l'ordre — ne pas optimiser sur les checks historiques

**E0 :** sur une période de développement autorisée, compter les tendances,
replis et reprises par actif et par mois. Vérifier invariance future,
géométrie M5, trous, non-chevauchement de signaux et maturités.

**E1, baseline figée :** comparer le taux de bougies suivantes haussières
sur les événements V1 au taux observé sur toutes les bougies en tendance
haussière éligibles de la même période. Afficher les supports, les
résultats par crypto, fold et régime et l'incertitude par blocs de jours.
Ce comparateur est descriptif : fréquence et distribution diffèrent.

**E2, seulement si E1 mérite exploration :** construire un modèle simple
(logistique régularisée) qui classe les événements V1. Features t uniquement :
écart close/EMA9 en ATR, écart EMA9/EMA26 en ATR, pente EMA9 3 barres,
profondeur du low précédent, corps en ATR, ATR relatif, rendement passé
3 barres. LightGBM peut être ajouté comme comparaison prédéfinie.
Entraîner sur train seulement, sélectionner sur validation seulement,
puis une unique comparaison chronologique sur des périodes non utilisées
pour concevoir les règles ; purge sur la maturité **maximale** des labels.

**E3 :** évaluer séparément les markouts retardés et leurs coûts théoriques.
Ne pas confondre précision directionnelle et rentabilité. Évaluer frais,
latence, slippage, absence d'exécution, ordres non remplis et barrières
intrabar ambiguës avant toute qualification économique.

## Hygiène expérimentale

Les 36 checks P2 et les anciens tuning ont été inspectés de nombreuses fois.
Tout résultat obtenu dessus sera marqué **EXPLORATOIRE ET CONTAMINÉ PAR LA
RECHERCHE ANTÉRIEURE**. Ne pas rouvrir ADA, les tests OOS historiques, le
final90 ou P3 pour départager les versions ; il faut des données futures
réellement nouvelles et un protocole gelé *avant* leur observation.
Un modèle perdant est archivé sans retoucher rétroactivement les seuils.

La branche n'ajoute aucun scheduler, WebSocket, exécution d'ordres ni
collecte prospective. La validation indépendante reste à organiser après
avoir arrêté les choix de recherche.

## Vérification du premier incrément

```powershell
cd E:\cryptoML-walkforward-test
git fetch origin
git switch research/ema9-26-pullback-v1
cd backend
python -m pytest tests/test_ema_pullback.py -q
python -m pytest -q
```

Les six tests du nouveau module ont été vérifiés sur fixtures synthétiques ;
cela ne signifie pas que l'intégration sur P0 réel ou toute la suite du
dépôt a déjà été validée.
