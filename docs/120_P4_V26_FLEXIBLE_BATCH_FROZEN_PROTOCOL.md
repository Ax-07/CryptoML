# P4 V26 — Micro-lots paramétrables, à validation explicite

**Protocole gelé le 22 octobre 2026, avant résultats V26.** Historique P4 déjà inspecté : BTCUSDT/ETHUSDT/SOLUSDT/XRPUSDT, 2024-10 à 2026-06 inclus, 84 partitions. P0 intouchable, juillet–septembre 2026 et ADA exclus. Aucun calcul de rendement, entraînement ou ordre.

## Référence vérifiée

V25 : 6 partitions validées après micro-lot ETHUSDT 2024-10 et SOLUSDT 2024-10 ; 78 manquantes, aucun échec signalé lors de l'exécution. V21 BTCUSDT 2025-06 ; V23 ETHUSDT 2025-06 et BTCUSDT 2025-05, 2024-10. **Ces 6 partitions restent les seules certifiées par leur registre local** et leurs archives SHA256.

## Deux phases strictes

**PLAN** : mode hors réseau, réexamine les 84 partitions via le registre V24 et valide les empreintes V22 des 336 HEAD. L'opérateur fournit **une ou deux cibles explicites** au format `SYMBOL:YYYY-MM`, appartenant à P4, distinctes, actuellement `MISSING`. Le pilote V21 `BTCUSDT:2025-06` n'est jamais réingéré. Le plan sérialise l'inventaire initial, le choix gelé, les chemins sources et les empreintes V22. Fichier de sortie **nouveau** et hors P0.

**EXECUTE** : nécessite `--permit-downloads`, un plan déjà enregistré, un SHA256 de ce plan calculé et confirmé par l'opérateur via `Get-FileHash`, et deux **nouveaux chemins** (journal JSONL et bilan JSON). Avant le moindre GET :
- comparer le SHA256 fourni à celui du plan et vérifier son schéma / univers / nombre de cibles ;
- valider les chemins protégés et comparer les empreintes actuelles V22 aux valeurs du plan ;
- réauditer les 84 partitions, refuser si une seule est `FAILED_INTEGRITY` ;
- n'accepter pour les cibles que `MISSING` ou `VERIFIED_V23`. Une cible désormais `VERIFIED_V23` est simplement journalisée `ALREADY_VERIFIED_SKIPPED`.

Pour chaque cible manquante (maximum deux), journaliser `STARTED`, effectuer **une seule invocation séquentielle** du moteur V23 (quatre ZIP + quatre CHECKSUM, schémas, SHA256 et comparaison P0), puis vérifier immédiatement la nouvelle partition via V24. Journaliser `VERIFIED` si toutes les validations passent ; sinon `FAILED` et interrompre la boucle sans toucher aux partitions préexistantes. Le journal est flush+fsync par événement.

Le bilan doit être écrit **même lors d'une exception gérée** de récupération/parser, avec le statut d'erreur, le nombre de partitions ajoutées, les données existantes intactes, le nombre `FAILED_INTEGRITY` du registre final, les SHA du plan/journal et les limites explicites. Une interruption brutale peut laisser un journal sans bilan, auquel cas créer de **nouveaux chemins** et relancer le même plan : il réaudite les 84 partitions et ignore celles déjà vérifiées. Ne pas masquer ni réparer automatiquement un répertoire invalide.

## Premier exemple figé

Plan 2 cibles **XRPUSDT:2024-10** et **ETHUSDT:2024-11** (distinctes des six précédentes). D'abord exécuter les tests et obtenir le **plan hors réseau**, inspecter l'affichage, puis décider séparément si les GET sont autorisés. Nombre attendu avant les GET : 6 verified / 78 missing ; après deux partitions effectivement conformes : 8/84. Ces nombres sont des attentes, pas une garantie.

## Contrôles et limites

- La présence HTTP 200 ne certifie jamais l'intégrité d'une archive.
- Les sources spot/perp/mark sont des OHLC M5, pas du bid/ask exécutable ; le funding réalisé n'était pas connu à la décision précédente. Ne jamais forward-fill les taux réalisés comme prédiction.
- Pas d'exécution automatique des 78 partitions ; l'utilisateur autorise explicitement chaque micro-lot.
- Après constitution des données et vérification de la provenance, **préenregistrer séparément** le modèle de risque/capital/liquidation/frais des deux jambes B01 carry et B02 basis. Garder un état NO_TRADE et exiger une validation prospective indépendante avant tout déploiement.
