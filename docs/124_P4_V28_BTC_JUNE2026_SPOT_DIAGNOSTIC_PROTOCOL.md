# P4 V28 — Diagnostic ciblé archive spot BTCUSDT juin 2026 (préenregistré)

**Geler avant toute nouvelle récupération de données.** État constaté au 23 octobre 2026 : 80 partitions vérifiées (1 V21, 79 V23), 4 `MISSING`, 0 `FAILED_INTEGRITY`. Une tentative V27 de juin 2026 s'est arrêtée **sur le premier actif BTCUSDT**, avec `Incorrect M5 count/shape` avant la ligne `spot verified`. Le journal consigne `newly_verified: 0`. La cause n'est **pas établie** : archive incomplète, CSV de forme différente ou erreur de parsing sont des hypothèses à départager.

## Objectif

**Aucune nouvelle ingestion des quatre derniers couples, aucun contournement V23/V24.** Le diagnostic V28 télécharge uniquement la **ZIP spot BTCUSDT 2026-06** et son fichier `.CHECKSUM` public. Il vérifie SHA256 avant ouverture, applique les limites ZIP et CSV du parseur V20, puis compare le contenu aux règles de V23 pour une grille UTC continue M5 sur le mois : **8640** bougies du 2026-06-01 00:00 UTC au 2026-06-30 23:55 UTC. Pour les timestamps, préserver la distinction secondes, millisecondes et microsecondes, avec l'unité réellement observée.

Rapport JSON **nouveau**, hors racine P0 et hors répertoire V23 immuable, sans écrire le ZIP, les prix, des snapshots de marché, un nouveau manifeste ou un PnL. Éléments à consigner : URL exacte et empreintes de ZIP/CHECKSUM, nombre de lignes CSV brut et hors en-tête, largeur CSV, présence d'un en-tête, première/dernière date, unité d'epoch, nombre de barres manquantes/dupliquées/hors mois, 10 premiers timestamps manquants au plus, statut de `parse_kline` et type de l'erreur.

## Interprétation sans anticipation des résultats

- **ZIP SHA invalide** : STOP, aucune conclusion de complétude.
- **ZIP SHA valide + bougies manquantes ou décalées** : l'archive d'origine n'est pas conforme au contrat 8640 M5. Ne pas inventer/imputer des bougies, ne pas modifier P0, ne pas affaiblir V23/V24. Examiner une source alternative *préspécifiée* et sa provenance séparée ; documenter les limites d'usage économique avant toute nouvelle ingestion.
- **ZIP SHA valide + 8640 M5 complets mais largeur/en-tête inattendu** : analyser schéma en détail, proposer une compatibilité ciblée soumise à tests.
- **ZIP SHA valide + M5 totalement conformes** : vérifier les différences éventuelles de version d'archive ou de diagnostic avant de reprendre. Ne pas conclure rétroactivement que le lot V27 était valide.
- **ZIP SHA valide, M5 complets, parseur V23 refusant une autre propriété** : traiter précisément cette propriété après un protocole correctif documenté ; jamais assouplissement silencieux.

L'historique jusqu'à juin 2026 reste **exploratoire, déjà inspecté**, pas d'OOS indépendant. Aucune simulation B01/B02, aucun ordre ou entraînement dans cette étape.
