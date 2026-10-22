# Phase 4 — Context-Aware Stealth Injection

État vérifié le 22 octobre 2026. Phase 4 terminée pour le projet original `attack-engine`.
Le développement et l'inspection portent sur ce workspace et ses documents locaux ;
aucun autre dépôt Attack Engine n'a été consulté pour cette phase.

## Fichiers et intégration

| Fichier | Responsabilité |
|---|---|
| [stealth.py](src/attack_engine/attacks/stealth.py) | Transformation séquentielle, projections, faisabilité quantifiée, contraintes et impact |
| [calibration.py](src/attack_engine/attacks/calibration.py) | Features causales, ridge par station, validation et sérialisation du petit artefact |
| [context.py](src/attack_engine/attacks/context.py) | Contexte en lecture seule, segments existants, historique des émissions antérieures |
| [models.py](src/attack_engine/attacks/models.py) | Champ optionnel `AttackRequest.stealth_context`, contexte typé et motifs de rejet |
| [config.py](src/attack_engine/attacks/config.py) | `StealthParameters` Pydantic, cinq familles exécutables, chemin relatif au YAML |
| [registry.py](src/attack_engine/attacks/registry.py) | Dispatch de `stealth` par la même interface que les quatre autres familles |
| [executor.py](src/attack_engine/attacks/executor.py) | Lecture de calibration, exécution chronologique, commits réels, résumé et couverture |
| [base.py](src/attack_engine/attacks/base.py) | Extension facultative des diagnostics de rejet ; politique numérique Phase 3 conservée |
| [cli.py](src/attack_engine/cli.py) | Commande explicite `calibrate`, génération en mémoire et présentation des diagnostics |
| [attacks.yaml](configs/attacks.yaml) | Paramètres expérimentaux explicites, sans modification de `scheduling.yaml` |
| [uscrn_train.json](calibrations/uscrn_train.json) | Artefact versionné, 8548 octets |
| [README.md](README.md) | Mode d'emploi, paramètres, unités, causalité et limites |

Tests ajoutés : [calibration](tests/test_context_calibration.py),
[transformation](tests/test_attacks_stealth.py), [exécuteur](tests/test_stealth_executor.py)
et [CLI](tests/test_stealth_cli.py). Fixtures et tests existants adaptés pour que Stealth
soit réellement exécuté ; les tests de la politique générique des familles non prises
en charge simulent maintenant une future famille non exécutable. Les régressions
numériques de Phase 3 et ses invariants Drift sont conservés.

Le scheduler, l'éligibilité, les sous-graines et le `BudgetLedger` existants sont réutilisés.
Aucune nouvelle dépendance ML n'est nécessaire. Les transformations Spike, Bias, Drift
et Replay gardent leurs mêmes paramètres et comptes sur la livraison réelle.

## Choix de modèle et calibration

Régression ridge standardisée indépendante pour chacune des six stations, via NumPy.
Le modèle est une référence contextuelle pour générer des faux signaux, pas le détecteur
IA final. Il ne consulte aucun score ou prédiction de ce dernier.

La sélection `train` précède le calcul des features. Les observations mesurées de 2022–2023
sont autorisées : timestamps de fin d'heure UTC tels que
`2022-01-01T00:00Z < time <= 2024-01-01T00:00Z`.
La dernière observation autorisée ferme donc la dernière heure de 2023.
Sur 105 120 lignes de train, **99374 lignes** disposent du contexte et de l'historique
requis pour ajuster les modèles. Aucun ajustement ni recalage des tolérances n'utilise
validation ou test.

Features, dans l'ordre enregistré :

1. `SOIL_MOISTURE_10`, `P_CALC` (mm cumulés dans l'heure), `SOIL_TEMP_5`, `T_HR_AVG`.
2. Dernière valeur cible observée avant l'heure courante.
3. Moyenne des 24 heures antérieures finies et contiguës.
4. Sinus/cosinus du cycle annuel et de l'heure UTC de fin d'observation.

Le modèle enregistre moyennes, échelles, intercept et coefficients. Une colonne constante
utilise une échelle de normalisation égale à 1. Le paramètre ridge vaut 1,0.
La tolérance est le quantile 0,99 des résidus absolus **dans le train**, avec plancher 0,001.
Il s'agit d'une enveloppe empirique ajustée sur les données de calibration, pas d'un seuil
d'alerte ayant un taux de faux positifs garanti. Les stations insuffisantes sont signalées
sans emprunter silencieusement un modèle d'une autre station.

| Station | Lignes train | Lignes utilisables | Tolérance résiduelle | Première ligne d'ajustement | Dernière ligne d'ajustement |
|---|---:|---:|---:|---|---|
| `AZ_Tucson_11_W` | 17520 | 17136 | 0.02600425 | 2022-01-02T01:00:00+00:00 | 2023-12-20T09:00:00+00:00 |
| `CA_Fallbrook_5_NE` | 17520 | 17466 | 0.00542316 | 2022-01-02T01:00:00+00:00 | 2024-01-01T00:00:00+00:00 |
| `MS_Newton_5_ENE` | 17520 | 17365 | 0.01375202 | 2022-01-02T01:00:00+00:00 | 2024-01-01T00:00:00+00:00 |
| `NY_Ithaca_13_E` | 17520 | 15752 | 0.01604177 | 2022-01-02T01:00:00+00:00 | 2024-01-01T00:00:00+00:00 |
| `OK_Goodwell_2_E` | 17520 | 14451 | 0.01645836 | 2022-04-24T00:00:00+00:00 | 2024-01-01T00:00:00+00:00 |
| `OK_Goodwell_2_SE` | 17520 | 17204 | 0.00762392 | 2022-01-02T01:00:00+00:00 | 2024-01-01T00:00:00+00:00 |

L'artefact contient la version de schéma `1.0`, la méthode, les bornes du train, la cible,
l'ordre des features, la profondeur d'historique, le pas temporel, les hyperparamètres,
les paramètres des stations, leurs périodes et effectifs, ainsi que les empreintes.

| Identité | SHA-256 |
|---|---|
| Calibration logique, JSON compact trié sans son propre champ de hash | `3a9700624450f00e0957a79d106045dbed49c458d30961a2a1ac5004e7e1e4a3` |
| Fichier JSON indenté réellement écrit | `2b758682269f7a102be34f99070b91e09b86417b38ab3096a823ce2c9dedea78` |
| Empreinte des seules entrées de train triées | `9ec4fb42d0d4ad058734c37e4a1b8361b71dd81d1ac4e89cc6ef80bafe1d4e9b` |

Le hash de train utilise les valeurs hashées par Pandas, sans l'index source ; les versions
de dépendances sont donc pertinentes pour le reproduire. Ce mécanisme identifie un contenu,
sans authentifier une origine physique. Le chargement contrôle le hash logique et la
compatibilité du schéma, des features, des hyperparamètres et des frontières de train.
La génération ne réentraîne jamais implicitement le modèle.

## Génération et causalité

La calibration est figée avant l'attaque. Stealth est autorisé sur validation/test seulement,
strictement après la fin de calibration. L'injection dans train est rejetée explicitement ;
aucun mode de calibration sur préfixe de train n'est annoncé ni implémenté.

L'exécuteur traite les fenêtres chronologiquement et conserve une copie des valeurs
effectivement émises. Le contexte précédant une fenêtre incorpore les modifications des
attaques antérieures de n'importe quelle famille ; les fenêtres futures sont exclues.
À l'intérieur d'une tentative, le lag et la moyenne utilisent les propres sorties
**déjà finalisées** de la tentative. Les résultats sont restitués dans l'ordre initial
du plan afin de préserver l'interface publique.

Pour chaque instant, la génération calcule `mu_t` depuis le contexte courant et l'historique,
puis projette `x_t + signe × magnitude_désirée` dans l'intersection suivante :

- Bornes physiques configurées, ici `[0 ; 0,70]`.
- Bande résiduelle `mu_t ± multiplicateur × tolérance_station`.
- Variation de la valeur émise limitée à `max_step_change × Δt`.
- Variation de l'offset limitée à `max_change_rate × Δt`.
- Variation totale cumulée des offsets limitée à `max_cumulative_drift`.
- Direction de perturbation configurée.

`Δt` vient des timestamps réels, y compris pour la première observation par rapport à
l'émission précédente. L'offset part de zéro au début de l'événement. La projection peut
modifier la magnitude réalisée par rapport à la magnitude souhaitée ; la variation cumulée,
les taux, le signe, les bornes et l'impact utile restent contrôlés explicitement.

Pour le profil réel : magnitude souhaitée 0,005–0,015 m³/m³, pente d'offset au plus
0,003 m³/m³/h, pente du signal au plus 0,05 m³/m³/h, variation totale d'offset au plus 0,08,
impact moyen absolu minimal 0,002 sur **toutes les observations réservées**.
Ces paramètres sont expérimentaux et ne constituent ni des seuils officiels ni un
calibrage contre les réponses d'un détecteur.

La grille 0,001 est prise en compte avant la finalisation partagée. Les valeurs finales
respectent les contraintes ; seuls des écarts d'arrondi float64 sont admis aux comparaisons
de calcul, sans seuil absolu dans le masque de modification. La politique de préservation
exacte de `0.141` et des valeurs intactes hors grille reste celle de Phase 3.

Les contextes inconnus ne sont pas imputés : pluie NaN n'est jamais une pluie nulle.
`reject_window` est le défaut ; `skip_observation` conserve exactement le propre pour
ces positions, l'inclut dans l'historique émis et comptabilise le retour de l'offset à zéro.
Un échec rejette l'intégralité de la tentative ; aucun préfixe partiel n'est committé.
Après huit tentatives au maximum, le dernier motif et la position d'échec sont rapportés.
Aucune contrainte ni partition n'est relâchée pour remplir le quota.

## Exécution réelle des cinq familles

Entrée : `../USCRN_livraison/data/uscrn_clean.parquet`, 210 384 lignes, 23 colonnes.
Graines de planification et génération : **14**.
Les fichiers `dataset.yaml` et `scheduling.yaml` restent ceux de Phase 3 ;
`attacks.yaml` ajoute le bloc Stealth documenté.

| Famille | Planifiées | Appliquées | Rejetées | Quota | Réservé | Plafond structurel | Committé | Quota non satisfait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spike | 534 | 534 | 0 | 1845 | 1845 | 1324 | 936 | 909 |
| bias | 70 | 61 | 9 | 3075 | 3051 | 3051 | 2798 | 277 |
| drift | 42 | 42 | 0 | 3074 | 2882 | 2840 | 2839 | 235 |
| replay | 75 | 75 | 0 | 2460 | 2427 | 2427 | 2416 | 44 |
| stealth | 49 | 22 | 27 | 1845 | 1767 | 872 | 690 | 1155 |
| **Total** | **770** | **734** | **36** | **12299** | **11972** | **10514** | **9679** | **2620** |

**Stealth :** 49 fenêtres et 1 767 positions réservées. Les 22 fenêtres appliquées réservent
690 positions, toutes effectivement modifiées. Les 27 rejets se répartissent ainsi :

- **22 fenêtres / 895 positions** : partition train non autorisée.
- **5 fenêtres / 182 positions** : intersection contextuelle/temporelle vide au dernier
  échec de tentative.
- **0 fenêtre ignorée** au titre d'une famille non implémentée.

La couverture des quatre mesures requises est **1 767 / 1 767 = 100 %** dans les fenêtres
Stealth planifiées pour cette graine, dont **872 / 872** dans les fenêtres validation/test.
Ce chiffre décrit la disponibilité de ces points précis. Il ne décrit pas la couverture
globale du dataset, une efficacité de détection ou un taux d'évasion.

Les paramètres réalisés ont aussi été vérifiés : impact moyen minimal parmi les événements
appliqués d'environ **0.005813 m³/m³** ; variation totale
d'offset maximale d'environ **0.015000** ;
pente d'offset maximale **0,003 par heure**, et pente de signal maximale **0,010 par heure**,
aux seuls arrondis machine près. Les contrôles sur USCRN recalculent les prédictions depuis
l'historique émis et vérifient la bande de résidus à chaque pas.

Réconciliation complète :

`12299 = 11972 réservés + 327 non réservés`

`11972 = 9679 modifications + 2293 réservations libérées`

`2620 de déficit = 327 non réservés + 2293 libérés`

Pour Stealth : `1845 - 690 = 1155 = 78 non réservés + 895 train + 182 infaisables`.
Le déficit est conservé et expliqué, sans redistribution. À la fin, aucune réservation
n'est active et `ledger.total_reserved == ledger.total_committed == 9679`.

Le plafond structurel exclut les fenêtres Stealth interdites dans train ; il demeure une
borne avant les rejets dépendant des données. La limite de capacité Spike mise en évidence
en Phase 3 demeure visible. Les quatre familles historiques gardent exactement
**936 / 2 798 / 2 839 / 2 416** modifications.

## Intégrité et répétabilité

Le SHA-256 de la source avant **et** après les générations est :

`46b98115424bb03af28d31bcb96b68fedd537a609bf97d4199a653a376b98421`

Deux plans frais et deux exécutions avec les mêmes entrées et paramètres sont identiques
en totalité : fenêtres, valeurs candidates, masques, diagnostics, références Replay et comptes.
Les comparaisons indépendantes confirment :

- Les 23 colonnes propres identiques avec `assert_frame_equal(check_exact=True)`.
- Les **5 401 cibles manquantes** préservées.
- Chaque masque égal à la différence finie exacte entre propre et final.
- **9 679 modifications uniques**, sans chevauchement.
- Le nombre de différences de la cible observée reconstruite en mémoire égal au ledger.
- Le hash du fichier source inchangé.
- Les totaux et les réservations de chaque événement et cellule réconciliés.

| Objet | Empreinte de comparaison |
|---|---|
| Plan, même sérialisation que dans l'audit Phase 3 | `b17ce2fbc438215125d060a7f08ed647a9b2897bb71d3eabddd7ddb99c26cb19` |
| `asdict(ExecutionSummary)`, JSON trié compact, dates ISO | `4fcf08b2585352ff757d3a94ae55e9b73ed6fa7ae0ff5a0354d852f6d06716ba` |

L'empreinte du plan est **inchangée par rapport à Phase 3**. Il ne s'agit pas de hashes
d'un dataset attaqué exporté : aucun fichier de ce type n'est créé.
L'identité est garantie ici dans l'environnement verrouillé utilisé, pas entre versions
arbitraires des bibliothèques.

## Vérifications exécutées

- `uv sync --locked` : succès, 20 packages résolus/vérifiés.
- `uv run ruff check .` : succès.
- `uv run ruff format --check .` : succès.
- Suite entière avec USCRN activé : **360 tests réussis**, aucun test réel ignoré.
- Quatre tests USCRN dédiés passent, dont calibration indépendante de toute modification
  des cibles et mesures futures de validation/test.
- Les tests synthétiques couvrent paramètres invalides, absence de contexte et de modèle,
  causalité des lags, gaps, dates de fin d'année, fit uniquement sur train, artefact altéré,
  quantification, bornes, impact, taux en temps réel, répétabilité, ledger et les cinq familles.

Un dossier temporaire Pytest dédié a été utilisé pour éviter le refus d'accès déjà
constaté en Phase 3 sur `%TEMP%/pytest-of-medma`. Les droits du dossier global et la
configuration de test du projet n'ont pas été modifiés.

Commandes de reproduction depuis `attack-engine/` :

```powershell
uv sync --locked
uv run attack-engine calibrate --input ../USCRN_livraison/data/uscrn_clean.parquet --config configs/dataset.yaml --attack-params configs/attacks.yaml --output calibrations/uscrn_train.json
uv run attack-engine generate --input ../USCRN_livraison/data/uscrn_clean.parquet --config configs/dataset.yaml --attack-config configs/scheduling.yaml --attack-params configs/attacks.yaml --json

$env:ATTACK_ENGINE_REAL_DATASET = (Resolve-Path ../USCRN_livraison/data/uscrn_clean.parquet).Path
$taskTestTemp = Join-Path $env:TEMP ('attack-engine-phase4-' + [guid]::NewGuid().ToString('N'))
uv run pytest -q --tb=short --basetemp "$taskTestTemp"
uv run ruff check .
uv run ruff format --check .
```

Environnement : Python 3.12.13, NumPy 2.5.4,
Pandas 3.1.0, PyArrow 26.0.0, Pydantic 2.14.1.
La génération écrit uniquement son résumé sur stdout. Seule la commande de calibration
écrit le petit artefact JSON, distinct des données originales.

## Limites et prochaine tâche

- Le modèle stationnel et ses résidus d'ajustement ne constituent pas une validation
  hydrologique complète. Des observations nouvelles peuvent rendre l'attaque infaisable.
- Les quatre mesures requises doivent être disponibles ; aucune profondeur structurellement
  absente ni station inconnue n'est remplacée automatiquement par un contexte inventé.
- Les rollouts utilisent des features causales, mais planification, validation d'une fenêtre
  complète et tentatives multiples restent un protocole hors ligne. Aucune efficacité
  d'attaque en ligne ou d'évasion du détecteur final n'est affirmée.
- Train n'est pas injecté par Stealth. Une future variante sur préfixe chronologique exige
  un protocole distinct ; elle ne fait pas partie de cette phase.
- Le signal peut revenir abruptement au propre après la réservation ; aucun point extérieur
  à la fenêtre n'est modifié pour lisser cette transition.
- Les budgets non réalisés restent rapportés. Une campagne centrée sur Stealth peut choisir
  explicitement un scope validation/test plutôt que de réserver aussi des fenêtres train.

**Prochaine tâche recommandée : Ground Truth, Provenance and Dataset Export.**
Elle devra conserver les masques exacts validés ici, isoler les features observables et
exporter de façon reproductible les résultats, la vérité terrain et la provenance.
Cette phase suivante n'est pas implémentée : aucun Parquet attaqué, fichier final de
vérité terrain, bundle de manifestes, détecteur, émulateur, signature, API ou dashboard
n'a été créé.

