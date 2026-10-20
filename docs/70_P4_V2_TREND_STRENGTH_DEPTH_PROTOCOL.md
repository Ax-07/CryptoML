# CryptoML P4 V2 — Régime de tendance et profondeur du pullback

**État : V2 descriptive uniquement. Aucun entraînement, aucun nouveau backtest, aucun signal promu.**

Branche `research/trend-pullback-regime-v2` créée à partir de la branche V1 figée (dernier commit V1 clôture `e6767e07ff5c6ca3d39d17e12b93219eaddd14d5`). La décision et les sorties de V1 restent documentées dans `docs/69_P4_V1_EXPLORATORY_CLOSEOUT.md`.

## Hypothèse V2 — définie avant consultation des étiquettes V2

Les caractéristiques de régime de tendance et de profondeur du premier retour vers EMA9 pourraient influencer la qualité d'un pullback. Cette hypothèse ne présuppose ni une direction ni un gain économique.

**Unité observée** : signal LONG retenu par le détecteur V1, strictement après clôture de la bougie t, sur M5. Aucun changement des conditions V1 de détection ni d'entrée n'est autorisé par cette étape.

Dix variables **calculées exclusivement à clôture(t)** :
1. Écart EMA9−EMA26 / ATR14 ;
2. variation EMA9 sur cinq M5 / ATR14 ;
3. variation EMA26 sur cinq M5 / ATR14 ;
4. croissance des clôtures t−12 à t−1 / ATR14(t) ;
5. profondeur entre EMA9(t) et le low(t), en ATR14 ;
6. profondeur entre EMA26(t) et le low(t), en ATR14 ;
7. corps signé open(t)→close(t) / ATR14 ;
8. mèche basse de t / ATR14 ;
9. position de clôture dans la fourchette low→high de t ;
10. volatilité ATR14 / close(t).

Aucune bougie après t. Les trous dans les données entraînent une remise à zéro de toutes les fenêtres. Les événements avec indicateurs incomplets restent comptés comme non classifiés, sans imputation depuis l'avenir.

### Catégories descriptives fixées à l'avance

À titre de **comptage uniquement** :
- Force : écart EMA9−EMA26 < 0.5 ATR (faible), de 0.5 à <1.5 (moyenne), ≥1.5 (forte).
- Profondeur : EMA9−low <0 ATR (sans sous-passage), de 0 à <0.5 ATR (peu profond), 0.5 à <1.0 (moyen), ≥1.0 (profond).

Ces valeurs ne sont pas des décisions de trading. **Ne pas rechercher, par comparaison rétroactive, le meilleur bucket**. Les catégories sont définies avant toute mesure de rendement V2.

### Phases prévues

1. Implémentation/test sur bougies **synthétiques** seulement : invariance des signaux passés si l'avenir change, gestion des trous, validité des indicateurs, aucun label consulté.
2. Recensement purement descriptif sur historique P0 **décembre 2025 à juin 2026** après vérification SHA-256, sans charger les rendements futurs pour sélectionner les buckets ; tous les résultats historiques sont déjà considérés comme du développement.
3. Étude éventuelle des labels et du score conditionnel : rédiger un protocole séparé et le geler avant exécution, avec purge à maturité et comparaisons à exposition identique, compte de coûts 2/5/10 bp et comparaison à zéro.
4. Si une nouvelle hypothèse/stratégie résulte de ces recherches : validation véritablement prospective après gel cryptographiquement horodaté des règles et modèles, observation des fills réalistes, frais, slippage et contraintes de portefeuille.

Ne consulter ni ADA réservée ni le segment juillet–septembre 2026 ni les fenêtres P3 ; aucun de ces historiques ne peut être requalifié silencieusement en OOS indépendant.

### Vérifier localement

```powershell
cd E:\cryptoML-trend-pullback
git status --short
git fetch origin
# Worktree propre seulement : rester en HEAD détachée si branche déjà attachée ailleurs.
git switch --detach origin/research/trend-pullback-regime-v2
cd backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_regime_v2.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

Aucun script V2 n'est autorisé à entraîner un modèle dans cette étape. Les tests doivent être validés sur Windows avant tout nouveau recensement.
