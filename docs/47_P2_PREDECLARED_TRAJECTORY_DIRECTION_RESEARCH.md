# P2 / CryptoML — Protocole figé de recherche directionnelle : trajectoire causale

**Statut : PROTOCOLE PRÉDÉFINI, NON EXÉCUTÉ — 19 octobre 2026.**  
**Branche :** \`feat/cpu-purged-walkforward-folds\`.  
**Priorité : H2 (10 min) ; H6 (30 min) uniquement en vérification secondaire.**

> **Important :** les checks P2 historiques ont déjà été consultés à de nombreuses reprises. Le choix de l'hypothèse et des nouvelles variables est informé par ces résultats : ce protocole est *préspécifié pour la prochaine expérience*, mais **ne rend pas rétroactivement les checks indépendants**. Aucune conclusion de généralisation ou de rentabilité ne peut en provenir.

## 1. Décision de recherche — lecture en 30 secondes

| Élément | Décision figée |
|---|---|
| Question | La forme du chemin parcouru et la position du cours dans une plage passée expliquent-elles la continuation ou la réversion, **au-delà des 30 features actuelles** ? |
| Résultat motivant H2 | LightGBM 34,08 % vs réversion M15 33,42 %, sur un masque commun choisi par LightGBM ; gain moyen **+0,658 point** sur 35 folds |
| Résultat motivant H6 | LightGBM 32,90 % vs réversion M30 32,44 % ; gain moyen **+0,459 point** |
| Décomposition des désaccords | H2 : 4 804 corrections ML, 4 345 corrections réversion, 5 639 désaccords NEUTRAL ; H6 : 5 513 / 5 188 / 6 750 |
| Actifs permis | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT ; **jamais ADA** |
| Modèle de référence | LightGBM CPU 3 classes, 30 features \`features_v1\`, maximum 150 estimateurs, seed 42, entraînement et early stopping inchangés |
| **Unique variante candidate** | 30 features + **deux** nouvelles features de trajectoire causale |
| Cible | \`vol_scaled_v1\`, DOWN/NEUTRAL/UP, multiplicateur **0,5** inchangé |
| Comparaison **principale** | H2, **mêmes lignes** sélectionnées par le modèle 30 features sur des scores de tuning, direction binaire ML vs direction binaire variante, NEUTRAL = erreur |
| Comparaison secondaire | Même procédure H6 ; logloss/Brier multiclasses brut et mouvement en diagnostics descriptifs |
| Succès exploratoire *minimum* | H2 : gain moyen **≥ +1,5 point**, **≥ 26 folds positifs sur 35** et gain positif dans **les quatre actifs** |
| Confirmation | Ne sera possible qu'avec des observations **chronologiquement nouvelles et autorisées** après gel complet ; **aucun** usage des tests OOS scellés, du dernier holdout 90 j, d'ADA ou de P3 |
| Actions interdites | Pas de nouvelle recherche de seuils/features sur les mêmes checks, pas de P&L ni trading réel, pas de promotion/relance des modèles figés |

**Attention au critère de succès :** +1,5 point et 26/35 sont des **seuils pratiques de tri prédéfinis**, et **non** un test statistique valide, car les folds et cryptos sont dépendants. Le respect de ces seuils ne justifiera qu'une **demande de confirmation future**, jamais un déploiement.

## 2. Inventaire — ce qui est déjà présent et déjà essayé

Les 30 entrées canoniques de \`backend/mlpipeline/features.py\` comprennent notamment :
- rendements 1, 3, 6, 12, 24, 48 M5 ; écarts EMA 12/48 ; RSI 14 ;
- volatilité, ATR, volume, nombre de transactions, taker buy share ;
- forme de la **bougie courante** (\`close_location\`, mèches/corps/range) ;
- calendrier et contexte BTC/ETH.

Le dépôt possède **déjà** \`market_structure_features.py\` et une expérimentation \`market_structure_diagnostics.py\` avec **12 variables** (HH/HL/LH/LL, BOS/CHoCH, distances et âges des swings). Ne **pas** les présenter comme de nouvelles variables ni recommencer aveuglément le test des 42 colonnes.

Les deux nouvelles variables ci-dessous décrivent un **chemin glissant de 12 bougies** et non un nouveau pivot confirmé. Leur complémentarité empirique reste **à démontrer** : elles peuvent être fortement corrélées aux rendements, EMA, RSI ou distances de swing déjà testés.

## 3. Hypothèse causale figée et définitions exactes

À chaque clôture M5 \`t\`, utiliser **uniquement** les bougies finales de \`t\` et du passé, au sein d'un **segment M5 continu** ; les fenêtres redémarrent strictement après un trou de données. Aucun usage de \`t+1\`, du label, de swings non confirmés, ni de l'identité d'actif comme feature.

### Feature F1 — \`trajectory_signed_efficiency_12\`

Mesure si le déplacement net s'est construit de façon régulière ou par de nombreux allers-retours.

Pour \`l_i = log(close_i)\`, à la clôture \`t\` :

\`F1(t) = (l_t - l_(t-12)) / Σ_(i=t-11..t) |l_i - l_(i-1)|\`

- 12 retours log 5 min, **60 minutes**, depuis \`t-12\` jusqu'à \`t\` ;
- valeur dans [-1, 1] pour un chemin normal ; positive si déplacement net haussier, négative si baissier, proche de zéro si chemin oscillant ;
- si la variation absolue totale est **exactement nulle**, retourner \`0\` ; avant 12 variations consécutives, \`NaN\` (warm-up).

**Différence avec \`ret_12\` déjà présent :** même déplacement net, mais normalisé par la somme des trajets intermédiaires. L'information supplémentaire est donc la *persistance du chemin*, pas un rendement de plus.

### Feature F2 — \`trajectory_prior_range_position_12\`

Position de la dernière clôture par rapport à la **plage des 12 bougies strictement antérieures** :

\`Lprev(t) = min(low_(t-12), ..., low_(t-1))\`  
\`Hprev(t) = max(high_(t-12), ..., high_(t-1))\`  
\`F2(t) = (close_t - Lprev(t)) / (Hprev(t) - Lprev(t))\`

- \`F2 < 0\` : clôture sous la plage ; \`F2 > 1\` : clôture au-dessus ; valeurs entre 0 et 1 : position interne ;
- pas de clipping ni recherche de fenêtre alternative ;
- si la plage historique a une largeur nulle, retourner **0,5** (convention déterministe) ;
- avant 12 bougies antérieures consécutives, \`NaN\`.

**Différence avec \`close_location\` existant :** celui-ci décrit la position du close **dans sa seule bougie**. F2 concerne l'étendue des **12 précédentes**, potentiellement cassée par la bougie courante. Il existe toutefois un possible recouvrement avec des features de swing précédemment testées ; ne pas supposer une nouveauté d'information.

### Gouvernance des nouvelles features

- Noms, formules, période 12, conventions de zéro et absence de clipping sont **figés** avant scoring.
- Une seule variante \`baseline_plus_trajectory_32\` (deux colonnes additionnelles) est autorisée pour cette phase ; **aucune** ablation supplémentaire individuelle, aucune durée 6/24, aucun nouveau groupe, aucun réglage de seuil sur les checks.
- Les colonnes canoniques \`FEATURE_COLUMNS\` et \`FEATURE_SCHEMA_VERSION\` ne changent **pas** ; les colonnes supplémentaires appartiennent uniquement à un module isolé de diagnostic.

## 4. Invariants de construction et sécurité

1. \`E:\cryptoML\data\` : **lecture seule**, partitions originales P0, contrôle des empreintes. Pas d'écriture dans \`E:\cryptoML\` ni de lancement du Python de production.
2. Checkout expérimental \`E:\cryptoML-walkforward-test\backend\`, Python \`E:\cryptoML-walkforward-venv-p3\Scripts\python.exe\` et résultats sous \`E:\cryptoML-walkforward-results\`.
3. Plan P2 faisant autorité : **36 folds** ; 120 jours train, 7 jours early stop, 3 jours tuning score-only, 4 jours development check ; label-maturity purge du **H12, 60 min** ; couverture cible 40 % sur tuning. Fold déjà infaisable historiquement conservé comme infaisable, **pas remplacé**.
4. **Pour chaque fold individuellement : tronquer les bougies à sa propre fin de check avant de créer les labels.** Les fonctionnalités causales ne lisent jamais de bougies de l'avenir.
5. Aligner strictement baseline/candidat sur les **mêmes** lignes train, early, tuning, check et les **mêmes** labels. Toute perte de lignes supplémentaires du candidat au-delà du warm-up déjà couvert par \`features_v1\` = erreur de protocole, pas suppression silencieuse. Assert effectifs contre le rapport historique.
6. N'évaluer **aucun** fold OOS test, dernier 90 jours, ADA ou sortie de modèles figés P3 / H6 prospective. Aucun flux en direct.
7. Les tests unitaires sur **données synthétiques** doivent prouver : invariance des préfixes, non-influence des barres futures, redémarrage sur trous M5, calcul aux bornes, convention pour volatilité nulle, absence de fuite des labels et équivalence des index d'échantillons. Une erreur interdit le smoke.
8. Écrire un rapport JSON nouveau sans écraser les anciens ; inclure SHA des données/plan, référence du commit, définitions figées et garde-fous \`no_reserved_scoring\`.

## 5. Comparaison mathématiquement appariée — protocole non négociable

### Principal : H2, évaluer la **direction sur le même masque**

- Entraîner les deux LightGBM **séparément mais à hyperparamètres et classes identiques**, avec le **même** train et le **même** early stop. Le candidat a seulement deux colonnes en plus.
- Construire sur les seuls scores de la tranche **tuning** du modèle **baseline 30** le masque d'abstention cible 40 %, avec ajustement \`alpha=1\` et marge top-deux préexistants. **Aucun label tuning/check** pour sélectionner le masque.
- Appliquer ce **masque baseline immuable** aux deux modèles sur le check. La couverture réelle peut dévier de 40 % ; **enregistrer** la couverture. Si tuning non faisable, conserver \`null\` pour les deux modèles sur ce fold.
- À chaque ligne sélectionnée, forcer le signe UP si \`p_UP > p_DOWN\`, sinon DOWN ; en cas d'égalité exacte, utiliser la direction majoritaire du **train** (à égalité DOWN).
- **NEUTRAL est une erreur**, pour les deux. Même masque, même dénominateur. Les comptes \`candidate_only_correct\`, \`baseline_only_correct\`, \`both_correct\`, \`both_wrong_NEUTRAL\`, \`both_wrong_OPPOSITE\` doivent se réconcilier avec les gains. Une catégorie « both wrong » peut avoir des prédictions différentes et être incorrecte pour le même label NEUTRAL ; journaliser séparément les désaccords.
- Métrique principale : **moyenne par fold** de \`100 × (précision_32 − précision_30)\` en points, sur les folds faisables et appariés. Enregistrer aussi le gain **pondéré par les lignes** ; ne pas mélanger les deux.

### Secondaires, sans promotion ni sélection de variantes

- Exactement le même diagnostic **H6** ; aucun changement du label ni des features.
- Score multiclasses sur **toutes** les lignes check : log loss, Brier multiclasses, accuracy/balanced accuracy (si déjà disponibles). Les probabilités ne sont pas les marges \`alpha=1\` : ne pas les confondre.
- Détection de mouvement : AP et ROC-AUC de \`1-p_NEUTRAL\`, Brier skill vs prévalence de mouvement **du train** ; direction conditionnée aux vrais mouvements = **diagnostic oracle seulement**.
- Comparaison secondaire avec \`reversion_3\` (H2) et \`reversion_6\` (H6), **sur le même masque baseline**, jamais une sélection oracle. Ces deux règles furent choisies après inspection d'autres checks.
- Résultats par actif et par fold seulement à des fins de robustesse descriptive. Les régions LOW/MID/HIGH peuvent être rapportées **sans optimiser de filtre** ; si incluses, seuils quantiles calculés sur train.
- Calculer les variations de couverture seulement pour des comparaisons supplémentaires où chaque modèle règle son propre seuil sur ses **scores tuning**. Elles ne remplacent pas la comparaison principale à masque commun.

### Critère de poursuite prédéfini

Considérer que le candidat *mérite d'être gelé pour une confirmation ultérieure* **seulement si** :
1. H2, delta moyen fold **≥ +1,5 point** de précision sur les mêmes décisions ;
2. H2, **au moins 26 des 35 folds** faisables ont un delta positif ; si le nombre de folds faisables diffère de 35, **suspendre** pour examiner la cause au lieu d'adapter le seuil ;
3. H2, delta **strictement positif sur chacune des quatre cryptos** ;
4. La logloss multiclasses H2 moyenne n'est pas dégradée de plus de **+0,005** (valeur candidat moins baseline) et aucun invariant sécurité/équivalence n'est violé.

Ces seuils ne démontrent aucune significativité statistique. En cas d'échec, consigner \`exploratory_negative_or_inconclusive\` et **ne pas** tester d'autres fenêtres, transformations ou filtres sur les mêmes checks pour sauver l'hypothèse. Quel que soit le résultat, **aucune promotion / trading**.

## 6. Étapes permises et suivantes

| Phase | Action | Autorisation actuelle |
|---|---|---|
| **P0 — Présent document** | Inventorier, fixer 2 features et une seule variante, critères et sécurité | **OUI : réalisé** |
| P1 — Tests purement synthétiques | Implémenter les calculs de F1/F2 isolés ; prouver la causalité, les segments M5 et les zéros | **Permis sans données historiques** |
| P2 — Smoke historique | 1 fold de développement uniquement, valider alignements/comptes/fingerprint, **pas choisir de features sur son score** | Après tests verts |
| P3 — Développement exploratoire | Répliquer exactement les 36 checks préexistants, comparer baseline30 vs variante32 | Seulement après smoke et revue de sécurité |
| P4 — Lecture une fois | Mesurer les critères **préfixés**, consigner les résultats même négatifs, figer le candidat s'il y a un signal | Pas de réglages rétrospectifs |
| P5 — Confirmation indépendante | Une fenêtre **chronologiquement nouvelle**, autorisée et scellée **après** gel ; protocoliser maturité, frais, exécution et tests pertinents | **Pas disponible/autorisé ici actuellement** |
| P6 — Envisager le réel | Revue sécurité/risque/exécution uniquement après preuve indépendante ; processus distinct de P3 | **NON** |

## 7. Interprétation et traçabilité

Les chiffres historiques de motivation proviennent des audits \`docs/42\` à \`docs/46\`, particulièrement \`p2_directional_rule_benchmark_full_v1.json\` et \`p2_directional_disagreement_full_v1.json\`. L'étude par régime observait des gains **positifs mais faibles** dans 20/24 cellules crypto/régime ; ce résultat ne fournit pas une règle de filtrage validée.

**À ne pas inférer :** une précision de 35 % sur une cible ternaire ne se traduit pas en 35 % de trades profitables ; AP de mouvement et précision directionnelle sont différentes ; gain sur des checks répétés ≠ preuve indépendante ; sélection oracle des vraies impulsions ≠ filtre déployable. Ni P&L ni coûts d'exécution n'ont été validés dans ce protocole.

**Livrable attendu après P2/P3 :** rapport comprenant par H2/H6 le nombre de folds appariés, précision candidate/baseline sur le masque partagé, delta moyen fold et pondéré, nombre de folds positifs/négatifs/égaux, ventilation par actif, logloss, détection, garde-fous sécurité et verdict \`experimental_only\`.

*Le document doit rester figé pendant l'expérience ; toute nouvelle hypothèse doit obtenir un nouveau numéro de protocole avant un éventuel nouveau cycle, pas être substituée silencieusement.*
