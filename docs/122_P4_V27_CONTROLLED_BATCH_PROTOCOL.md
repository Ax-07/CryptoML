# CryptoML P4 V27 — Ingestion par lots bornés (1 à 4), autorisation vérifiable

**Protocole figé le 22 octobre 2026, avant exécution V27.**

## Point de départ vérifié

L'opérateur a confirmé V26 : tests ciblés et suite complète réussis ; plan SHA256 validé ; archive XRPUSDT 2024-10 et ETHUSDT 2024-11 vérifiées en séquence ; total du registre **8 partitions sur 84**, **76 manquantes**, sans échec déclaré. V22 HEAD avait affiché 336/336 URL potentiellement accessibles ; **seules 8 partitions sont réellement certifiées SHA et contre P0**.

## But et limites

Passer de lots de 2 à **au plus 4 couples actif/mois** par approbation explicite, sans boucle automatique sur les 76 restants. Les traitements peuvent être repris après interruption, mais jamais en écrasant une partition. Les données restent uniquement P4 : BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, octobre 2024 à juin 2026. ADA et juillet–septembre 2026 exclus ; l'historique a déjà été examiné, **pas OOS indépendant**.

V27 ne calcule **aucun** basis, carry, funding PnL, signal, modèle ML, ordre ou gain théorique.

## Deux modes

- **`plan` (0 GET/HEAD)** : l'opérateur renseigne 1 à 4 cibles `SYMBOL:YYYY-MM` uniques. Le programme vérifie les 84 partitions (via V24 : fichiers ZIP, CHECKSUM, Parquets, grilles UTC, funding et parité P0) ; arrête au moindre `FAILED_INTEGRITY` ; confirme que chaque cible est `MISSING` ; valide la couverture V22 complète ; consigne les sources, empreintes SHA V22, cibles explicites, et **SHA des manifestes initialement vérifiés** dans un nouveau plan JSON, hors P0 et hors datasets.
- **`execute`** : demande `--permit-downloads` + SHA256 exact du plan autorisé + nouveaux chemins journal JSONL et bilan JSON. Revalide empreintes V22, fichiers P0/racines et les SHA des manifestes initialement vérifiés ; arrête avant le réseau si une partition est `FAILED_INTEGRITY`. Pour chaque cible : si déjà `VERIFIED_V23`, écrit `ALREADY_VERIFIED_SKIPPED` ; sinon vérifie un **plancher de 512 MiB d'espace disque libre** puis lance exactement une ingestion V23 (4 ZIP + CHECKSUM, contrôle SHA, UTC, comparaisons P0) et ré-audite le résultat. Les téléchargements sont **séquentiels**, jamais parallèles.
- **Traçabilité** : journal `PRECHECK_OK`, `STARTED`, `VERIFIED`, `ALREADY_VERIFIED_SKIPPED`, `FAILED`, `BATCH_DONE` ou `BATCH_FAILED`, un fsync par événement, digest du manifeste sur chaque succès. À la première anomalie, **STOP** ; écrire un bilan même si l'ingestion a échoué proprement. Une interruption brutale peut laisser journal partiel et pas de bilan ; reprise avec **même plan et nouveaux noms de journal/bilan**, après nouvel audit.
- Tout résultat `HEAD_OK_UNVERIFIED` reste uniquement un indice de disponibilité. Aucune réparation de ZIP/P0 ou contournement automatique d'une anomalie.
- Le plan **n'autorise que les quatre cibles exactes**. Un nouveau lot requiert un nouveau plan et un nouvel accord via son SHA256.

## Premier essai

Plan figé : `BTCUSDT:2024-11`, `SOLUSDT:2024-11`, `XRPUSDT:2024-11`, `ETHUSDT:2024-12`, aucun filtrage par retours/financement. C'est une sonde de robustesse d'ingestion sur quatre partitions non exploitées. Succès attendu **12/84 vérifiées et 72 manquantes**, à confirmer uniquement après GET et contrôle SHA.

**Avant B01/B02 :** validation temporelle multi-source exhaustive, funding effectivement payé à sa date, deux jambes spot/perp exécutables avec spread/slippage, marge/risque liquidation, rebalancement, coût capital/borrow et NO_TRADE ; protocole économique gelé avant lecture du PnL. Aucun essai sur données P4 déjà inspectées ne peut prétendre être indépendant.
