# P4 V24 — Registre vérifié et ingestion reprise maîtrisée

**Gel du protocole : 22 octobre 2026, avant toute exécution de V24 sur l'historique.**

V21 : BTCUSDT juin 2025, archives spot/perp/mark/funding checksum validées, 8 640 bougies et 90 funding. V23 : ETHUSDT juin 2025, 7 tests ciblés et suite entière réussis, quatre sources validées, 8 640 M5 et 90 funding. V22 : 336 / 336 URL HTTP 200, sans validation de ZIP.

## Périmètre

Quatre actifs (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT), 21 mois 2024-10 à 2026-06 inclus, exactement **84 partitions symbol/mois**. Quatre sources par partition, soit 336 ZIP/checksums potentiels. **Juillet–septembre 2026 restent exclus**, tout résultat sur les 21 mois reste exploratoire et non OOS.

Aucune opération de trading, aucun calcul de PnL, aucune sélection de signaux ou entraînement. V24 doit uniquement savoir **quelles partitions sont véritablement réutilisables** et autoriser une extension contrôlée.

## Deux modes séparés

**AUDIT (défaut, zéro réseau)** :
- Recenser les 84 couples et marquer `VERIFIED_V21`, `VERIFIED_V23`, `MISSING` ou `FAILED_INTEGRITY` ;
- pour les partitions existantes V21 et V23, vérifier `manifest.json`, les quatre ZIP + leurs CHECKSUM SHA256, les hash des deux Parquets, les colonnes OHLC et la continuité M5 complète, les funding datés séparément, et la concordance perp/P0 ;
- ne jamais déduire un succès d'un dossier ou d'un manifeste seul ; un dossier incomplet est `FAILED_INTEGRITY`, et interdit de lancer une ingestion ;
- l'unique chemin historique V21 connu est passé explicitement et est **lecture seule** ; le V23 root est également lecture seule en mode audit ;
- produire un nouveau CSV et un JSON (84 lignes et agrégats) hors du P0, **aucun téléchargement**.

**INGEST (réseau opt-in, traitement ciblé)** :
- lancer un audit complet d'abord ; si une partition est corrompue, **stopper avant tout GET** ;
- imposer `--symbol`, `--start-month`, `--max-new-partitions` (1 ou 2), `--permit-downloads`. Choisir au maximum deux mois manquants de cet actif, à partir du mois donné et dans le calendrier gelé. Les partitions déjà vérifiées sont **ignorées sans modification**, jamais écrasées ;
- valider la couverture V22 complète, puis réutiliser `ingest` V23 pour chaque nouvelle partition : quatre ZIP, quatre CHECKSUM, SHA, parsing, funding et comparaison P0 avant publication atomique. Aucun parallélisme de GET. À la première erreur, **STOP** ; conserver les partitions déjà publiées et les journaux d'erreur ;
- après l'opération, refaire un audit et produire un nouveau rapport ; les sorties d'audit sont toujours nouvelles. Aucun lot de 82 partitions implicite.

## Contrôles de sécurité et reproductibilité

1. Les racines P0, V21 et V23 sont des chemins résolus. Output V23 et sorties rapports obligatoirement hors P0. Chemins de rapport distincts, inexistants. Interdiction des écritures dans le répertoire V21.
2. Les SHA256 des ZIP et des .CHECKSUM doivent correspondre au manifeste et entre eux. Les Parquets doivent correspondre à leurs SHA manifest. P0 : manifeste qualité OK, SHA et lignes concordants, OHLC perp identiques.
3. Dates M5 UTC complètes, closes exclusives, OHLC positifs et géométrie cohérente, volume spot/perp non négatif et mark sans volume négocié. Funding : taux en fraction décimale, horodatages uniques croissants dans le mois, intervalles 1–24h ; ne pas imposer 90 événements hors du mois pilote.
4. P0 demeure strictement immuable et **aucune donnée future réservée n'est lue**.
5. `HEAD_OK_UNVERIFIED` demeure un simple indice de disponibilité, jamais une preuve d'intégrité ou de profit.
6. Le succès de deux partitions ne valide pas encore les 82 autres. Aucun arbitrage funding/basis ne sera simulé avant l'intégrité multi-sources et un protocole économique deux jambes gelé.

## Première exécution prévue

1. AUDIT des deux partitions présentes. Résultat visé : 1 VERIFIED_V21, 1 VERIFIED_V23, 82 MISSING, 0 FAILED, **à confirmer sur le PC de l'opérateur**.
2. Puis **1 nouvelle partition au maximum** : BTCUSDT 2025-05 (renseignée explicitement), jamais une boucle automatique sur 84.
3. Si succès, inspecter le manifeste, la couverture et les limites de stockage avant d'envisager un lot de 2 partitions.
