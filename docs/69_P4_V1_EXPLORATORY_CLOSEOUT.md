# P4 V1 — Clôture de la recherche exploratoire (20 octobre 2026)

**Décision : NE PAS PROMOUVOIR. Pas de stratégie prête au trading.**
**Gel méthodologique : les événements EMA9/EMA26, baselines A/B, entrées, stops,
labels, seuils et Ridge P4 V1 restent inchangés. Ne pas améliorer V1 à l'aide
des résultats déjà consultés.**

Ce document archive les **sorties de console transmises par l'opérateur**
le 20 octobre 2026. Les JSON/CSV complets, leurs empreintes et les Parquets
restent sur son disque local et n'ont pas été fournis dans cette conversation ;
ce n'est donc pas un audit indépendant des fichiers ou de leur provenance.

## Résultats observés

- Suite `pytest -q` du worktree : pas d'échecs signalés ; seulement des avertissements
  de dépréciation Starlette/httpx et LightGBM.
- Période P0 décembre 2025–juin 2026, quatre paires BTC/ETH/SOL/XRP.
  Détecteur : **6 814** configurations, réparties 1717 / 1729 / 1716 / 1652.
- Baseline A : **6 813 événements mûrs** ; moyenne nette théorique
  **−1.6673436296706594 bp par opportunité à 2 bp**.
- Baseline B (confirmation bougie t+1) : **2 885 trades** parmi 6 813 événements ;
  couverture **0.42345515925436666** ; net par opportunité :
  **−0.5099671992133993 bp à 2 bp**,
  **−1.7803326769764993 bp à 5 bp**,
  **−3.8976084732483325 bp à 10 bp**.
  À 5 bp, seulement 2 des 28 groupes paire-mois B sont positifs,
  ETH et XRP en février 2026.
- Ridge C V1 : StandardScaler + Ridge(alpha=10) et seuil final de score
  net théorique à 5 bp **0.0** (`score > 0`), calibré sur avril.
  Diagnostics mai–juin : **84 trades Ridge** sur **825 trades B** (10.2 %).
  Ridge améliore le net/opportunité par rapport à B mais
  **6 des 8** groupes paire-mois ont un net/trade plus défavorable que B ;
  seul BTC juin (+5.276 bp/trade) et SOL juin (−0.588 bp/trade)
  dépassent le net/trade de B dans leur groupe.
- Comparaison descriptive de sélection à même allocation crypto-mois :
  l'espérance d'un tirage aléatoire de 84 trades parmi les B du même groupe
  correspond à environ **−5.67 bp/trade** ; Ridge a environ
  **−7.93 bp/trade** (différence environ **−2.26 bp/trade**).
  **Ce n'est pas une permutation exécutée ni une p-value**.
- La référence zéro trade = **0 bp** bat globalement A, B et Ridge C à 5 bp.

## Interprétation bornée

Résultats théoriques sur historique **déjà exploré** :
ni preuve OOS indépendante, ni capacité de généralisation, ni PnL
réel. L'OPEN futur t+2 est un prix de référence hypothétique et non
un remplissage garanti ; spread, slippage, capacité et ordres
concurrents ne sont pas modélisés de façon réaliste.

Le gain de B par rapport à A est surtout un gain de filtrage ;
Ridge réduit encore la fréquence, mais n'a pas démontré de meilleure
sélection nette par trade. Ne pas choisir XRP, février, juin BTC
ou les rares résultats positifs a posteriori pour bâtir une
preuve de robustesse.

## Verrouillage et suite

1. Ne pas retoucher les sources P4 V1 sur cette branche à partir de
   ces diagnostics. Garder les résultats JSON/CSV originaux immuables,
   idéalement calculer et conserver leurs SHA-256 côté opérateur.
2. **V2 sur branche distincte** : nouvelle hypothèse descriptive a priori
   sur force de tendance et profondeur du repli. Première phase : nouvelles
   features causales + tests synthétiques + comptages **sans labels**.
3. Tout apprentissage V2 ultérieur utilisera uniquement une partition de
   recherche annoncée et une spécification figée avant l'analyse des
   résultats ; les historiques déjà explorés restent exploratoires.
4. L'ADA réservée et les dates de validation antérieurement réservées
   ne deviennent pas automatiquement des « nouveaux » holdouts.
   Une validation confirmatoire exige des observations futures et un
   gel externe vérifiable de la politique avant leur collecte.

**Statut final P4 V1 : `RESEARCH_CLOSED_NO_ELIGIBLE_STRATEGY`.**
