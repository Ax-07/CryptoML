# P4 V18 — Mode opératoire : disponibilité des archives et du funding Binance

L'étude A06 V17 est close : brut +1,262191 bp par opportunité, mais **net −0,737809 bp à 2 bp** et −3,737809 bp à 5 bp sur 8 833 opportunités. Aucun changement rétroactif de seuil. [Clôture et protocole](104_P4_V18_A06_CLOSEOUT_FUNDING_BASIS_FEASIBILITY.md).

## Installation

```powershell
cd E:\cryptoML-volume-economics-v17
git fetch origin
git worktree add E:\cryptoML-funding-basis-v18 origin/research/funding-basis-source-feasibility-v18
cd E:\cryptoML-funding-basis-v18\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest tests/test_funding_basis_source_probe_v18.py -q
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m pytest -q
```

**STOP si tests en échec** ; code publié sans prétendre à une exécution locale sur le PC de l'opérateur.

## Échantillon public (opt-in ; aucun ZIP téléchargé)

Le programme ne contacte pas Binance sans le flag explicite `--probe-public-sources`. Avec flag : **12 HEAD** au maximum sur `data.binance.vision` (BTCUSDT, 4 catégories, mois 2024-10 / 2025-06 / 2026-06) et **1 GET public** `/fapi/v1/fundingRate` sur juin 2025. Aucun token, clé API, ordre ou position. HEAD 200 signifie présence probable du fichier, **pas** qu'il est valide, complet ou checksum vérifié. HEAD 404/403/451 signifie non confirmé ; les CDN peuvent répondre différemment aux HEAD. La sonde de funding ne donne qu'un échantillon de paiements réalisés, pas une stratégie à information causale.

```powershell
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.funding_basis_source_probe_v18 `
  --probe-public-sources `
  --output-json "E:\cryptoML-walkforward-results\p4_v18_funding_basis_sources_v1.json"
```

Sortie toujours **nouvelle**, JSON hors P0. Si les archives recherchées sont disponibles, la prochaine étape sera un protocole d'**ingestion** contrôlée de spot+perp+funding et de checksum de chaque mois sur l'univers choisi, puis la pré-déclaration du modèle du financement et des deux jambes. **Aucun résultat économique ni arbitrage rentable ne sera déduit de ce test de disponibilité.**

Aucune activité live automatique requise.
