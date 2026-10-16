# CryptoML P2 — Diagnostic de confiance directionnelle (CPU)

## Objectif et limites

Sur les 36 validations de développement déjà étudiées, précision directionnelle moyenne 32,24 %, référence aléatoire tenant compte des fréquences 27,92 %, avantage +4,31 points, couverture 59,44 %. **Ces validations ne sont plus indépendantes**, car elles ont déjà guidé la recherche. Les 90 derniers jours et ADA ont également été consultés lors de l'évaluation finale précédente et sont désormais interdits comme jeux de sélection.

Cette étape étudie une hypothèse : certaines décisions UP/DOWN sont-elles plus fiables lorsque le score gagnant dépasse nettement le second score ? Il ne s'agit pas d'une promesse de performance.

Le protocole conserve :
- **LightGBM CPU, 30 features, horizon H6 = 30 minutes** ;
- **alpha fixe à 1.0** pour toutes les fenêtres, comme référence préalablement figée, sans retuning ;
- 120 jours d'entraînement ; dans les 14 jours de validation : **7 jours early stopping, 3 jours choix du seuil, 4 jours vérification** ;
- 14 jours OOS par fold **non évalués** et 90 derniers jours **non évalués**, ADA **exclue des folds**.

Le score de confiance est la différence entre les deux meilleurs **scores de décision normalisés après correction par les priors**. **Ce ne sont pas des probabilités calibrées.** Les seuils candidats sont : 0 ; 0,01 ; 0,02 ; 0,03 ; 0,05 ; 0,075 ; 0,10 ; 0,15 ; 0,20 ; 0,25 ; 0,30.

L'algorithme choisit sur les 3 jours de réglage le seuil qui maximise **l'avantage de précision directionnelle par rapport à une référence aléatoire tenant compte des fréquences**, sous contraintes de **couverture au moins 20 %** et **au moins 50 prédictions de chaque direction**. Si aucune candidate ne respecte ces contraintes, repli explicite sur seuil 0, signalé comme tel.

Ensuite, le seuil est appliqué **sans aucun réglage supplémentaire** sur les 4 jours de vérification. Le rapport conserve précision, couverture, erreurs NEUTRAL, rappel et précision par direction, référence aléatoire sur toutes les classes et référence conditionnelle sur la sélection. Les résultats sont également ventilés par symbole BTC/ETH/SOL/XRP.

**Attention :** l'ajustement de la confiance peut simplement isoler des marchés avec davantage de mouvements directionnels. La comparaison à la référence conditionnelle aide à distinguer ce cas d'un véritable avantage prédictif. Aucune de ces métriques ne prouve une performance exploitable en trading, et les labels M5 glissants ne sont pas indépendants.

## Exécution Windows PowerShell

Depuis `E:\cryptoML-walkforward-test\backend` :

```powershell
& "E:\cryptoML\backend\.venv\Scripts\python.exe" -m mlpipeline.confidence_diagnostics \
  --data-root "E:\cryptoML\data" \
  --months 24 \
  --horizon 6 \
  --train-days 120 \
  --validation-days 14 \
  --test-days 14 \
  --step-days 14 \
  --final-holdout-days 90 \
  --n-estimators 150 \
  --verify-hashes \
  --output "E:\cryptoML-walkforward-results\confidence_24m_p2.json"
```

Dans PowerShell, **remplacer chaque `\` de continuation par le caractère backtick `\`** (ou saisir la commande sur une seule ligne). Le script refuse d'écraser un rapport existant.

Il ne crée aucun ordre ni aucune position et ne modifie ni les fichiers P0, ni les modèles figés. Le résultat attendu est un JSON contenant les 36 folds, les seuils ajustés sur tuning seulement, ainsi que la précision/couverture sur les contrôles chronologiques.

## Interprétation

Une hausse de précision accompagnée d'une baisse extrême de couverture n'est pas une solution. Priorité aux **différences de précision sur les fenêtres de vérification**, à la constance entre les folds et aux tailles d'échantillon DOWN/UP, puis à la comparaison aux références conditionnelles. Toute amélioration retenue est encore une hypothèse de développement : elle devra être évaluée prospectivement sur de nouvelles données non consultées.
