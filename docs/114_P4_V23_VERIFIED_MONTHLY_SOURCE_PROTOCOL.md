# P4 V23 — Protocole gelé : ingestion immuable par partition spot/perp/mark/funding

Date de gel : 21 octobre 2026, avant toute ingestion V23.

## Faits confirmés

- Les 3 tests V22 et la suite entière passent sur la machine de l'opérateur.
- V22 échantillon : 48 HEAD 200 / 48 ; V22 complet : **336 HEAD 200 / 336**, couvrant `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT`, octobre 2024 à juin 2026 inclus, quatre types `spot,perp,mark,funding`.
- HEAD ne prouve pas l'intégrité : V21 a **vérifié les quatre archives BTCUSDT juin 2025** par SHA256, construit 8640 M5 communes et 90 événements funding, et comparé les bougies perp au P0. **Tout autre mois-actif reste non certifié.**

## Objet V23

Créer un extracteur **par partition** (`--symbol`, `--month`) ne traitant **qu'un couple actif/mois par invocation** (quatre ZIP et quatre CHECKSUM), puis valider intégralement avant publication dans une arborescence nouvelle, immuable, distincte du P0. Le code est conçu pour les 84 partitions, mais la première exécution est **ETHUSDT juin 2025**, hors du pilote V21, afin de confirmer le schéma sur un deuxième actif. **Pas d'exécution automatique de 336 téléchargements en V23.** Les 84 peuvent être traitées progressivement après tests réels et budget disque/réseau.

## Garde-fous et traçabilité

1. Entrées verrouillées à quatre actifs et 21 mois. `--coverage-json` + `--coverage-csv` V22 complets obligatoires ; toutes les 336 lignes uniques doivent correspondre au protocole exact et aux 336 HEAD 200 annoncés. Une URL peut avoir changé entre HEAD et GET.
2. Une seule partition par commande, **réseau opt-in**. Télécharger exactement quatre ZIP publics de Binance Data Vision, plus quatre CHECKSUM. Aucun accès compte, aucune clé ni ordre ; protéger les tailles compressées, décompressées et les membres ZIP.
3. Revalider SHA256 en GET de chaque ZIP vs son CHECKSUM officiel avant parsing. Staging temporaire dans output root ; destination unique `symbol=.../month=...` refuse l'écrasement, les chemins P0 et une racine d'output dans P0.
4. Normaliser explicitement secondes / millisecondes / microsecondes à partir des longueurs uniformes d'epoch. Spot µs depuis 2025 ; selon source ne pas supposer une unité unique pour les quatre catégories. Une bougie M5 n'est disponible qu'après open+5 min, indépendamment des close times Binance inclusifs.
5. Spot/perp/mark : **M5 exactes et intégrales du mois**, OHLC >0, volumes spot/perp >=0, mark sans volume fictif ; `close_time` fournisseur vérifié lorsque présent. Strict alignement des trois marchés, aucun remplissage de trous.
6. Funding : archive distincte, timestamps d'événement UTC strictement croissants uniques et dans le mois, champs `calc_time`, `last_funding_rate`, `funding_interval_hours` explicitement validés, taux en fractions finies, intervalle 1–24h. **Ne pas imposer 90 événements/mois :** fréquence et longueur du mois variables ; ne jamais forward-fill funding vers des bougies.
7. Comparer sur la même partition mensuelle le M5 perp au P0 immuable : SHA256 du Parquet et manifeste P0, qualité OK, timestamps et OHLC identiques à tolérance numérique stricte ; aucune modification P0.
8. Sorties : quatre ZIP + CHECKSUM bruts et deux Parquets (`markets_5m.parquet`, `funding_events.parquet`), manifeste SHA256 de tous les fichiers, SHA256 du rapport V22 complet et CSV, empreinte SHA du P0 mensuel. Publication seulement quand tout passe ; pas de revendication de couverture 84/84 avant validation de chaque partition.
9. Utiliser uniquement les mois historiques déjà étudiés, pas les mois de juillet-septembre 2026 réservés ou d'autres périodes, **aucune prétention OOS**. Aucun rendement, entraînement, arbitrage, performance spot/perpétuel, benchmark capitalisé ou ordre en V23.

## Décision après pilote ETHUSDT juin 2025

Si les 4 archives et la comparaison P0 passent, V23 fournit **deux partitions certifiées indépendamment** avec V21. Une étape ultérieure peut gérer reprise multi-partition, registre consolidé de SHA et inventaire d'échecs, puis un protocole de simulation delta-neutre sans lookahead : événements de funding connus seulement à leur date, deux jambes avec frais, bid/ask, slippage, risque basis/liquidation, marge et contraintes d'exécution, contrôle NO_TRADE et validation prospective future. Si une archive diffère, ne pas supprimer le contrôle pour « faire passer » : enregistrer la cause exacte et garder le dossier P0 inchangé.
