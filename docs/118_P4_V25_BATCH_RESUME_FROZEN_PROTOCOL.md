# P4 V25 — Plan immuable, exécution bornée et reprise contrôlée des partitions multi-sources

**Protocole gelé le 22 octobre 2026, avant toute exécution V25.** V24 confirmé par l'opérateur : BTCUSDT 2024-10, 2025-05, 2025-06 et ETHUSDT 2025-06 vérifiés ; **4/84** partitions multi-sources, **80 MISSING**, **0 FAILED_INTEGRITY**. Les URL des 336 ZIP avaient retourné HEAD 200 en V22, mais ne sont pas 336 archives SHA vérifiées. Pas de backtest de funding/basis ni de training.

## Intention

Automatiser uniquement un **micro-lot explicite de 1 ou 2 couples actif-mois maximum**, sans effacer de partitions existantes. Séparer strictement deux phases :

- `plan` (**aucun réseau**) : auditer les 84 couples en réutilisant `registry` V24 : SHA des ZIP/checksums et Parquet, OHLC et funding, comparaison P0 ; contrôler la couverture V22 complète ; arrêter au moindre `FAILED_INTEGRITY`. Sélectionner au maximum deux couples `MISSING` dans un mois approuvé et dans une liste explicite d'actifs. Sauvegarder un **nouveau JSON de plan** hors P0, incluant chemins de sources, empreintes V22, état initial et choix fixés.
- `execute` (**opt-in strict**) : accepter **uniquement un plan valide** issu de `plan`, `--permit-downloads` explicite et un nouveau journal JSONL + un nouveau JSON de bilan ; re-auditer les 84 partitions AVANT tout téléchargement, rejeter les partitions en erreur, revérifier les SHA V22 et les racines. Pour chaque couple gelé : si déjà `VERIFIED_V23`, enregistrer `ALREADY_VERIFIED_SKIPPED` (aucune écriture) ; si `MISSING`, journaliser `STARTED`, invoquer exactement une ingestion V23 (quatre ZIP+quatre CHECKSUM, vérification et comparaison P0) et **revérifier** la partition avant de poursuivre. Arrêter au premier échec et journaliser `FAILED`. Aucun retrait de garde-fou pour accélérer.

## Garanties et limites

1. Univers strict : quatre actifs P4 (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT), 21 mois 2024-10 à 2026-06, ADA exclue, juillet–septembre 2026 exclus. Pas de données futures.
2. Les dossiers V21 (BTC 2025-06), V23 et P0 sont en lecture seule en mode plan. En execute, seules les nouvelles partitions V23 peuvent être créées ; V21 et P0 jamais modifiés. Dossiers déjà existants ne sont jamais écrasés.
3. Revue d'intégrité V24 sur l'**ensemble** du registre avant toute nouvelle requête réseau. Une partition incomplète interdit toute ingestion jusqu'à correction manuelle documentée.
4. Le journal est créé exclusivement comme **nouveau fichier** ; chaque événement est flush et fsync sur disque, pour diagnostiquer une interruption. Le bilan terminal est produit après succès/échec si le processus reste vivant. La mort brutale du processus peut empêcher le bilan terminal, mais le journal déjà écrit et les partitions promues subsistent ; une **nouvelle exécution nécessite de nouveaux chemins de journal et bilan**, après réaudit. Pas de suppression / réutilisation silencieuse.
5. Les **deux candidats** maximum sont fixés dans le plan, pas sélectionnés à nouveau au fur et à mesure des résultats. Toute modification d'empreinte V22 ou des chemins source bloque l'exécution.
6. Pour l'essai initial, les cibles sont **ETHUSDT 2024-10 et SOLUSDT 2024-10**, choisies avant inspection de leurs archives par GET : cela vérifie une ancienne période sur deux actifs supplémentaires. Aucun choix en fonction du funding/rendements.
7. Aucun parallélisme de GET, aucune ingestion des 80 d'un coup, aucun arbitrage ou rendement, aucun entraînement ML ou ordre. `HEAD_OK_UNVERIFIED` n'est pas une validation SHA.
8. L'historique P4 est **déjà inspecté** ; même une ingestion complète ne créera pas une validation OOS indépendante. Un protocole économique deux jambes sera préenregistré plus tard, avec funding à la date de paiement, bid/ask, frais, slippage, marge, liquidation, rééquilibrage, coûts de capital et NO_TRADE.

## Contrôles d'acceptation

- `plan` : rapport initial 4 verified / 80 missing / 0 failed, deux couples planifiés, fichiers de sortie immuables ;
- `execute` : réaudit avant GET, puis deux partitions V23 réellement contrôlées / aucune anomalie ; état cible **6 verified / 78 missing** si les deux archives nouvelles sont conformes.
- En cas d'échec : **ne pas promettre 6/84** ; analyser le journal et arrêter les GET. Aucune création de signaux ni PnL.
