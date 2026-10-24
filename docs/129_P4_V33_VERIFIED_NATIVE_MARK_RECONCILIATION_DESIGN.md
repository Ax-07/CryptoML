# P4 V33 — contrat de conception, PAS une ingestion autorisée

## État gelé
Audit V24 indépendant : 80/84 vérifiées, 4 MISSING, 0 FAILED_INTEGRITY. Pour juin 2026 sur BTCUSDT, ETHUSDT, SOLUSDT et XRPUSDT, les archives `futures/um/monthly/markPriceKlines` contiennent chacune 8352 des 8640 bougies attendues, avec exactement les 288 bougies du 29 juin absentes. Les archives `futures/um/daily/markPriceKlines` du 29 juin fournissent 288 M5 par actif. V30 et V32 attestent indépendamment des archives mensuelles et quotidiennes vérifiées par CHECKSUM, absence de chevauchement, alignement UTC, validation V23 par journée et réunion temporelle complète. Ces observations ne signifient PAS quatre partitions V23 vérifiées.

## Contrat d'ingestion à développer séparément
- Conserver les 80 partitions existantes, la racine P0, les V21/V23 et les plans/journaux V27 intacts.
- N'autoriser explicitement que les quatre couples (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT) × 2026-06, en lot borné avec journaux et rapports exclusivement nouveaux.
- Geler l'identité des archives SHA256 individuelles déjà constatées, puis revérifier, au téléchargement, chacun des cinq ZIP et CHECKSUM de chaque partition : spot monthly, perp monthly, mark monthly, mark daily 2026-06-29, funding monthly. Ne pas considérer SHA comme une preuve de complétude.
- Construire les 8640 mark M5 en fusionnant exclusivement les 8352 M5 natifs du mensuel et les 288 M5 natifs du quotidien. Ne pas extrapoler, interpoler, resampler ou substituer BUSD/spot/perp au mark price.
- Vérifier pour chaque marché toute la forme, géométrie OHLC, cadence M5, horaires inclusifs de clôture, prix positifs, absence de doublons, cohérence des 8640 timestamps et alignement spot/perp/mark. Vérifier funding et comparaison immuable avec P0.
- Conserver **dans le dossier de partition de sortie** les 5 ZIP originaux et 5 CHECKSUM, et inscrire dans un manifeste versionné distinct les SHA256, URLs, rôles source, frontières temporelles, compte exact de barres par source (8352/288) et empreintes des Parquet produits.
- Choisir un dossier de destination racine **distinct de V23**, immuable et inexistant pour chaque partition, et définir un auditeur indépendant compatible V33 qui vérifie aussi le contenu des ZIP, l'union des timestamps et l'identité des prix reconstruits (ne pas faire confiance à la seule présence d'un manifeste).
- Publier une partition seulement après revalidation déterministe complète, via staging dans la même racine suivi d'un renommage atomique sans écrasement. Si une source manque/change ou échoue, n'écrire aucune partition cible et conserver un journal nouveau du refus.
- Faire évoluer explicitement le registre agrégé : 1 V21 + 79 V23 + jusqu'à 4 V33. Ne pas afficher `84 VERIFIED_V23` ou supposer qu'un audit V24 inchangé comptera les V33.
- Ne jamais prétendre qu'il s'agit d'un OOS indépendant, ni lancer PnL, ordres, entraînement ou B01/B02 à cette étape.

**Statut : implémentation V33 expérimentale et tests synthétiques publiés sur branche isolée ; aucune exécution sur archives réelles, aucun résultat d'audit V33 confirmé et AUCUNE autorisation de publier avant revue/tests complets.**
