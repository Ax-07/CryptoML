# P4 V4 — Audit descriptif de l'économie mécanique A/B

**Statut : implémenté, tests synthétiques à confirmer localement.**
Aucune stratégie nouvelle, aucun entraînement, aucun réglage et aucun OOS indépendant.

La branche `research/trend-pullback-economics-audit-v4` dérive du résultat V3 clôturé dans `docs/78_P4_V3_RIDGE_EXPLORATORY_CLOSEOUT.md`.
Elle est séparée des branches V1/V2/V3 et leurs paramètres sont inchangés.

## Mesures fixées avant l'exécution

- Tous les pullbacks A mûrs et le sous-ensemble B confirmé au close(t+1), selon les mêmes règles t+2, stop 1 ATR, target 1.5 ATR, horizon six bougies et priorité au stop si deux barrières sont touchées.
- Rendement *brut* théorique bp : moyenne, médiane, écart-type, quantiles p01/p05/p25/p50/p75/p95/p99, comptages gagnants / nuls / perdants.
- Six motifs de sortie : `gap_stop`, `gap_target_capped`, `both_hit_stop_first`, `stop`, `target`, `timeout`.
- Décomposition comptable des coûts **2, 5, 10 bp par aller-retour** ; invariants exacts net=brut−coût.
- Effectifs, couverture et net/trade + net/opportunité sur l'ensemble, chacun des 4 actifs et chaque mois **2024-10–2026-06**.
- Comparaison descriptive à la référence sans transaction (=0), **sans chercher les meilleurs mois, symboles, sorties ou seuils**.

Chargeur : exactement 84 Parquets P0 avec SHA-256 vérifié pour BTC/ETH/SOL/XRP, sans ADA ni juillet–septembre 2026. Les données restent les mêmes recherches internes déjà examinées ; résultats non indépendants. Les rendements et fills restent hypothétiques et excluent la dynamique réelle d'exécution.

## Exécution Windows

```powershell
cd E:\cryptoML-trend-pullback-v3
git fetch origin
git worktree add E:\cryptoML-pullback-economics-v4 origin/research/trend-pullback-economics-audit-v4
cd E:\cryptoML-pullback-economics-v4\backend

& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_pullback_economics_audit_v4.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q

# UNIQUEMENT si les deux commandes de tests sont vertes :
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.pullback_economics_audit_v4 `
  --data-root "E:\cryptoML\data" `
  --output-json "E:\cryptoML-walkforward-results\p4_v4_ab_economics_audit_v1.json" `
  --output-csv "E:\cryptoML-walkforward-results\p4_v4_ab_economics_audit_v1.csv"
```

Les sorties sont **nouvelles** et le script n'écrase pas les anciennes. Le JSON final est écrit après le CSV. Les tests sont synthétiques et ne chargent aucune donnée réelle.

## Arrêt méthodologique

Ce diagnostic vise à décrire pourquoi les trades B affichent, en mai–juin 2026, un net moyen négatif à 5bp. Ne pas déduire des sous-groupes rentables de ces étiquettes une règle exploitable : il faut une hypothèse nouvelle documentée avant toute autre évaluation et, pour une conclusion, une validation véritablement prospective.
