# P4 V14 — Préenregistrement du comparatif économique multi-stratégies A01–A05

**Statut : PROTOCOLE ÉCONOMIQUE GELÉ AVANT OUVERTURE DES RETOURS V14.** V13 a produit 6 211 signaux sans labels (A01=823, A02=2 270, A03=675, A04=409, A05=2 034). Ces comptes proviennent de la sortie de l'opérateur, pas d'une lecture directe des CSV/JSON de son PC. Les règles de signal et leurs paramétrages **ne changent pas**. Aucune optimisation, aucune exécution d'ordres, aucun entraînement ML.

## Question à mesurer

Chaque famille capture-t-elle un mouvement suffisant pour absorber des coûts aller-retour hypothétiques de 2 / 5 / 10 bp ? Les rendements positifs après ces coûts ne prouvent pas un edge futur : octobre 2024–juin 2026 (4 actifs BTCUSDT ETHUSDT SOLUSDT XRPUSDT) est **un échantillon déjà étudié**. Juillet–septembre 2026 et ADA demeurent exclus de P0 V14 ; la V10 live demeure arrêtée.

## Provenance et synchronisation strictes

Le chargeur V3 ouvre **exactement 84 Parquets P0** dont SHA-256, finalité, qualité et comptes sont vérifiés. Reprendre les **trois sorties V13** déjà produites ; requérir le JSON `p4_v13_multistrategy_census_v1.json` et le CSV `p4_v13_multistrategy_signals_v1.csv` pour audit. Avant d'évaluer des returns, revérifier empreinte manifests, univers, protocole, comptes par famille/actif/sens, unicité, timestamps, prix de référence et ATR. **Recalculer les signaux V13 sur les mêmes bougies pour vérifier identité événement par événement**, sans regarder d'outcomes. Toute divergence stoppe l'étude.

Signaux considérés à leur **`decision_close_at`** seulement. Entrée théorique sur le `OPEN` de la prochaine M5 consécutive ; ne pas supposer un fill executable à cette valeur. Rejeter toute entrée sans M5 suivante ou horizon complet, bougies discontinues, maturité atteignant le 1er juillet 2026 00:00 UTC, ATR non fini / <=0, stop/target non positif, prix incohérents. Conserver le décompte de **toutes** les exclusions.

## Gestion des positions hypotétiques : une configuration par famille, sans sélection opportuniste

Pour chaque entrée, stop LONG = `entry − ATR_known`, target LONG = `entry + 1.5 × ATR_known`. SHORT = miroir. `ATR_known` est l'ATR14 de la timeframe du signal H4 (A01), H1 (A02–A04), ou M5 gelé à la cassure (A05). **N'utiliser aucune ATR post-signal**. Vérifier cohérence avec les conventions V8 : si open traverse stop défavorablement, sortie à cet open ; si open traverse target favorablement, plafonner au target ; si les deux barrières sont touchées dans la même bougie, **stop prioritaire** ; sinon prix de barrière ; timeout au close de la dernière M5. Gross = `10000 × direction × (exit/entry − 1)`; net = brut moins coût aller-retour de 2 / 5 / 10 bp. **Pas de funding, spread variable, slippage, levier, marge, liquidation ou exécution réelle** : le résultat est un diagnostic optimiste, surtout pour le H4 48h.

**Horizon principal unique *par mécanisme*, arrêté d'avance** :
- A01 Momentum H4 : **576 bougies M5 = 48h** maximum ;
- A02 Donchian H1 : **144 bougies M5 = 12h** maximum ;
- A03 Compression H1 : **144 bougies M5 = 12h** maximum ;
- A04 Mean Reversion H1 : **48 bougies M5 = 4h** maximum ;
- A05 V9 Retest : **12 bougies M5 = 1h**, cohérent avec la primaire V11. **Exiger 24 barres M5 consécutives pour qualifier une opportunité A05**, comme la cohorte commune V11, même si seules les 12 premières sont utilisées pour le markout.
- Aucun horizon alternatif n'est testé ni choisi sur V14. Les horizons **ne sont pas comparables en fréquence, durée et risque** ; ne jamais choisir « le gagnant » à partir des seules moyennes nettes.

La décision peut produire plusieurs hypothétiques opportunités qui se recouvrent (sur un actif ou entre stratégies). Ces montants sont des **markouts de candidats**, pas un portefeuille et pas un rendement de capital. Aucun netting ni exclusion de conflits a posteriori pour doper les moyennes. Zéro trade reste le comparateur 0 bp et doit être explicite.

## Rapport obligatoire

JSON + CSV nouveaux **hors P0**, avec identité V13, empreinte SHA-256 CSV, comptes du sous-ensemble maturé, exclusion explicite par cause, par actif/stratégie/sens/mois. Pour chaque configuration : `trades`, moyenne/médiane brut et net/trade 2/5/10, taux net positif, somme brute et nette notionnelle, motifs de sortie, horizon primaire, temps de détention avant stop/target/timeout en minutes, décompte de chevauchements théoriques à qualifier. Aucune proportion « gain » inventée et aucun retour financier n'est déduit des nombres V13.

Préserver l'historique (no overwrite) ; CSV détaillé des événements + CSV mensuel si nécessaire. Rédiger une clôture exploratoire même en cas de zéro événement. Réconciliation des agrégats mensuels/symboles obligatoire avant publication.

## Décision et étape ML

Ne pas entraîner le méta-modèle à choisir le meilleur **retour futur observé**. La décision cible devra se fonder sur des variables *connues avant l'entrée*, incluant validité du signal, régime, volatilité, prix et coûts anticipables. Le portefeuille multi-stratégie doit être défini séparément (conflits, sizing, exposition, fréquences, financement et capital). Il faudra une période entièrement nouvelle et indépendante pour démontrer une sélection adaptative supérieure aux baselines fixes, avec correction du risque de sélection après plusieurs essais.

**V14 demeure `EXPLORATORY_ALREADY_EXAMINED_HISTORY` et `NO_TRADE` jusqu'à preuve contraire.**
