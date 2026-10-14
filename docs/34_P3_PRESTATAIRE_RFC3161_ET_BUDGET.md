# P3 — Sélection de prestataire d'horodatage : décision d'achat non prise

**État au 14 octobre 2026 : candidat identifié, NON CONNECTÉ, NON TESTÉ en production.**

## Candidat étayé par le site officiel

Certigna (France), service d'horodatage qualifié eIDAS utilisant des jetons RFC 3161 et horodatant des **empreintes SHA256** plutôt que des documents complets. Documentation et tarifs publics : https://www.certigna.com/famille-de-services/horodatage/

La page commerciale affiche : 100 jetons = 36 € HT, 500 = 89 € HT, 2 000 = 269 € HT, 10 000 = 809 € HT, packs valables un an, et une offre abonnement annuel sur demande. **Ceci n'est ni un devis ni une promesse de volume, de latence, de droits API ou de service 56 jours.** Confirmer tous les termes directement avant d'acheter.

## Volumes requis

- 5 min par bougie, 24 h/jour, 56 jours = **16 128 clôtures M5**.
- 4 actifs × 2 horizons prédits dans **le même enregistrement par clôture**. Théoriquement un seul jeton RFC 3161 par digest d'enregistrement (et NON un jeton par actif et par horizon).
- Minimum nominal par packs publics : deux packs de 10 000 jetons = **1 618 € HT** (3 872 jetons non consommés), *si la commande de deux packs et leur usage dans cette intégration sont permis*.
- Alternative purement arithmétique : 1×10 000 + 3×2 000 + 1×500 = 16 500 jetons, **1 705 € HT**. Le premier choix est moins cher sur ces prix publics.
- Les besoins réels incluent tests, interruptions éventuelles, reprise du service et provision ; requérir un devis commercial sur une capacité 17 000–20 000 jetons et un plan tarifaire alternatif.
- Ne pas déclencher la collecte tant que les contrats et performances ne sont pas établis.

## Incompatibilité de protocole — étape bloquante

Le client actuellement codé exige **JSON + signature Ed25519 + HTTPS GET par empreinte**. Certigna annonce **RFC 3161**, soit un jeton cryptographique signé (TSA/CMS) et des vérifications de certificat et d'empreinte. L'API existante **NE SAIT PAS** utiliser les jetons RFC 3161. Il faut une nouvelle interface de réception des jetons et une validation stricte par la chaîne de confiance fournie par la TSA, notamment le certificat de l'autorité, l'empreinte, l'algorithme, la politique, le temps UTC, le statut du jeton et l'attribution à la TSA.

Le protocole RFC3161 (https://www.rfc-editor.org/info/rfc3161/) exige la vérification du jeton signé et de la correspondance de l'empreinte. Un jeton RFC3161 conservé avec la prévision est **vérifiable hors ligne**, contrairement à un service GET propriétaire. Son stockage durable doit être indépendant du disque local : une archive append-only/WORM avec audit d'intégrité reste nécessaire pour la restitution et la preuve de complétude. **Ne pas inventer une URL API, un certificat racine, des identifiants ou une compatibilité Ed25519.**

## Audit préalable du candidat (à remplir)

- [ ] Compte et contrat commercial/abonnement ; autorisation d'usage pour 16 128 requêtes (+ marge)
- [ ] Endpoint RFC 3161 officiel et authentification fournis au compte ; débit et temps de réponse compatibles avec limite stricte de 270 s
- [ ] Certificat TSA, chaîne CA, liste de confiance et politique d'horodatage obtenus via canal officiel et testés
- [ ] Vérification *effective* d'un jeton RFC3161 et de l'empreinte incluse par client indépendant
- [ ] Archive externe WORM/append-only opérée séparément, preuve de persistance et droit d'export/audit
- [ ] Monitoring Windows: redémarrage, alertes, alimentation, horloge UTC/NTP indépendant, logs hors machine
- [ ] Staging sur **nouvel enregistrement** séparé, prédictions non scorées et conservation testée
- [ ] Nouvel enregistrement P3 prospectif enregistré **avant une nouvelle date de début**; ne pas utiliser celui du 16 octobre si le processus est incomplet

## Commande sans achats ni activation

```powershell
cd E:\cryptoML-walkforward-test\backend
& "E:\cryptoML-walkforward-venv-p3\Scripts\python.exe" -m mlpipeline.prospective_vendor_plan `
  --root "E:\cryptoML-walkforward-results\prospective_h2_h6_p3"
```

Le programme donne les volumes, déclare explicitement l'incompatibilité d'API et ne contacte pas le prestataire.

**Décision à ce stade : étudier Certigna pour RFC3161, ne rien acheter ou activer automatiquement.** Aucune évaluation des nouvelles étiquettes, ADA, ancien OOS, ni trading.
