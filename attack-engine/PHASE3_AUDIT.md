# Phase 3 — Audit de correction

Audit du 22 octobre 2026, limité à Spike, Bias, Drift et Replay. Aucun Stealth, export de
dataset, label de Phase 4 ni changement de l'algorithme de planification Phase 2 n'a été
ajouté. Le projet local ne contient pas de dépôt Git ; les fichiers de code, les
configurations et leurs empreintes ont été examinés directement.

## Conditions de reproduction

Entrée : `../USCRN_livraison/data/uscrn_clean.parquet`, 210 384 lignes, 23 colonnes,
204 983 cibles finies. Graines de planification et d'exécution : **14**. Configurations :
les trois YAML présents dans `configs/`, sans modification.

| Élément | SHA-256 |
|---|---|
| Dataset | `46b98115424bb03af28d31bcb96b68fedd537a609bf97d4199a653a376b98421` |
| dataset.yaml | `df5e66f83b186274131b7e39b98faf90f07544d61d182ee8046014577dc9eae2` |
| scheduling.yaml | `0a5447e03a0950724662549717e9ce5acc9cad25118b2c5fe0b584af77306792` |
| attacks.yaml | `71701009ee264ff674626d57ef870bfe7c978cc7d1e7c6be4c5915ce0d53e984` |

Environnement verrouillé utilisé : Python 3.12.13, NumPy 2.5.4,
Pandas 3.1.0, PyArrow 26.0.0, Pydantic 2.14.1.
Toutes les générations ont été effectuées en mémoire ; aucun Parquet attaqué n'a été écrit.

## Défauts confirmés et corrections

1. La quantification globale d'une candidate pouvait changer un point auquel aucune
   perturbation n'était demandée. Avec `clean = 0.141`, le produit
   `round(clean / 0.001) * 0.001` donne une autre représentation float64.
   Une comparaison exacte naïve le comptait comme modifié.
2. Le correctif présent au début de l'audit, `CHANGED_ATOL = 1e-9`, évitait plusieurs
   faux positifs mais supprimait aussi des attaques réelles plus petites que ce seuil.
   Il modifiait encore des observations intactes hors grille, par exemple `0.1414`.
   Il a été remplacé par la politique de quantification décrite ci-dessous.
3. Bias autorisait des dépassements de bornes avec un seuil fixe de `1e-9`, puis
   vérifiait la constance uniquement sur les positions modifiées. Une perturbation
   affectant seulement une partie de la fenêtre pouvait ainsi être acceptée comme
   biais constant. La tolérance est maintenant liée à l'arrondi float64 et le contrôle
   inclut toutes les positions.
4. La capacité Drift était surestimée si `hold_first=false`, alors que la rampe
   commence toujours à un temps relatif nul pour une requête valide. Le plafond
   structurel retranche désormais une position de chaque fenêtre Drift planifiée.
5. Une régression utilisait `result.window.length`, attribut inexistant. Elle utilise
   maintenant `request.window.length`, avec les assertions exactes demandées.
6. La commande `generate` pouvait poursuivre après un SHA-256 différent du hash
   configuré. Elle refuse maintenant cette entrée avant de planifier/exécuter.
7. Les paramètres numériques non finis sont rejetés à la validation de configuration ;
   le helper de quantification refuse aussi un quantum infini ou NaN.

Les mécanismes de politique explicite pour les familles non exécutables et de capacité
structurelle déjà présents ont été conservés et testés. Le résumé expose en plus
l'allocation, la part non réservée et le quota total non satisfait, en préservant les
champs existants. Aucune transformation du dépôt n'a été remplacée par une nouvelle
architecture.

## Politique numérique explicite

- **Aucune intention de changement** (`candidate == clean`) : conserver exactement le
  float original, qu'il soit sur la grille ou hors grille.
- **Quantum activé** : comparer les bins obtenus par arrondi au plus proche, moitié
  vers le pair. Si les bins de la candidate et du propre sont identiques, conserver
  exactement le propre. Quantifier la référence pour cette comparaison ne la réécrit pas.
  Une petite perturbation traversant une frontière de bin est conservée.
- **Quantum désactivé** : aucune tolérance de détection. Toute différence finie
  représentable, y compris une ULP, est une modification.
- Quantifier puis borner une candidate effective ; restaurer le propre sur les
  positions finalement inchangées. Calculer le masque par comparaison exacte entre
  les valeurs finales et propres finies. Le masque et les valeurs retournées
  décrivent ainsi la même réalité.
- Bias utilise `4 * eps_float64 * échelle` uniquement pour les vérifications
  d'arrondi arithmétique et de constance. Ce terme n'intervient jamais dans le masque
  ni comme seuil scientifique d'amplitude.

Les régressions couvrent notamment `0.141 / 0.001`, des valeurs hors grille intactes,
une attaque de `1e-12` sans quantum, une différence d'une ULP, une attaque de `1e-10`
avec quantum `1e-12`, une perturbation sous-quantum qui traverse un bin, les bornes,
les zéros effectifs, Bias partiellement modifié et Replay partiellement identique.
Une perturbation qui reste dans le même bin est supprimée par la résolution
**explicitement configurée**, pas par un epsilon absolu caché.

## Réconciliation du Drift annoncé

Une reconstitution en mémoire de l'ancien finaliseur — quantification de toute la
candidate, écrêtage et `final != clean` — reproduit exactement **2 843** modifications
sur **42 fenêtres / 2 882 positions réservées**. Quatre premières observations sont
alors faussement modifiées :

| Fenêtre | Premier propre | Ancien premier final |
|---|---|---|
| `drift-AZ_Tucson_11_W-train-0263` | 0.051 | 0.051000000000000004 |
| `drift-OK_Goodwell_2_E-train-0498` | 0.206 | 0.20600000000000002 |
| `drift-AZ_Tucson_11_W-validation-0592` | 0.036 | 0.036000000000000004 |
| `drift-CA_Fallbrook_5_NE-validation-0629` | 0.143 | 0.14300000000000002 |

Après correction :

`committed = 2 839 = 2 843 - 4`

`committed <= reserved - applied_windows = 2 882 - 42 = 2 840`

L'unique position inchangée supplémentaire est l'index local **1** de
`drift-NY_Ithaca_13_E-train-0447`, fenêtre de 238 observations :
la première marche de rampe vaut environ `0.1 / 237 = 0.00042194`, inférieure à un
demi-quantum de 0.001 pour ces valeurs. Elle reste dans le même bin. Les 43 réservations
Drift inutilisées sont donc **42 origines de rampe + 1 marche quantifiée sans effet**.

L'ancien calcul comptait aussi **1 038** modifications Spike ; la correction en compte
**936**, soit 102 faux changements éliminés. Bias reste à 2 798 et Replay à 2 416 pour
cette configuration. Le total ancien reconstitué de 9 095 devient **8 989**.

## Audit de chacune des 42 fenêtres Drift appliquées

La première candidate a été capturée à l'entrée réelle de `finalize_candidate`, avant
quantification ; la valeur finale et son masque ont été capturés à la sortie.
L'égalité des trois valeurs est vérifiée exactement, indépendamment de leur affichage
décimal. La longueur, le nombre réservé et le nombre réellement committé sont joints
par identifiant de fenêtre. Aucun premier masque n'est vrai.

| Fenêtre | Longueur | Propre[0] | Candidate brute[0] | Final[0] | changed_mask[0] | Réservé | Committé |
|---|---:|---:|---:|---:|---|---:|---:|
| `drift-AZ_Tucson_11_W-test-0031` | 81 | 0.02 | 0.02 | 0.02 | false | 81 | 80 |
| `drift-AZ_Tucson_11_W-test-0032` | 46 | 0.028 | 0.028 | 0.028 | false | 46 | 45 |
| `drift-CA_Fallbrook_5_NE-test-0066` | 85 | 0.245 | 0.245 | 0.245 | false | 85 | 84 |
| `drift-CA_Fallbrook_5_NE-test-0067` | 40 | 0.133 | 0.133 | 0.133 | false | 40 | 39 |
| `drift-MS_Newton_5_ENE-test-0105` | 123 | 0.297 | 0.297 | 0.297 | false | 123 | 122 |
| `drift-NY_Ithaca_13_E-test-0135` | 33 | 0.146 | 0.146 | 0.146 | false | 33 | 32 |
| `drift-NY_Ithaca_13_E-test-0136` | 30 | 0.126 | 0.126 | 0.126 | false | 30 | 29 |
| `drift-NY_Ithaca_13_E-test-0137` | 40 | 0.393 | 0.393 | 0.393 | false | 40 | 39 |
| `drift-NY_Ithaca_13_E-test-0138` | 25 | 0.11 | 0.11 | 0.11 | false | 25 | 24 |
| `drift-OK_Goodwell_2_E-test-0172` | 98 | 0.157 | 0.157 | 0.157 | false | 98 | 97 |
| `drift-OK_Goodwell_2_E-test-0173` | 24 | 0.164 | 0.164 | 0.164 | false | 24 | 23 |
| `drift-OK_Goodwell_2_SE-test-0204` | 100 | 0.243 | 0.243 | 0.243 | false | 100 | 99 |
| `drift-OK_Goodwell_2_SE-test-0205` | 25 | 0.213 | 0.213 | 0.213 | false | 25 | 24 |
| `drift-AZ_Tucson_11_W-train-0263` | 103 | 0.051 | 0.051 | 0.051 | false | 103 | 102 |
| `drift-AZ_Tucson_11_W-train-0264` | 42 | 0.046 | 0.046 | 0.046 | false | 42 | 41 |
| `drift-AZ_Tucson_11_W-train-0265` | 50 | 0.061 | 0.061 | 0.061 | false | 50 | 49 |
| `drift-AZ_Tucson_11_W-train-0266` | 49 | 0.047 | 0.047 | 0.047 | false | 49 | 48 |
| `drift-CA_Fallbrook_5_NE-train-0330` | 68 | 0.181 | 0.181 | 0.181 | false | 68 | 67 |
| `drift-CA_Fallbrook_5_NE-train-0331` | 174 | 0.271 | 0.271 | 0.271 | false | 174 | 173 |
| `drift-MS_Newton_5_ENE-train-0386` | 165 | 0.326 | 0.326 | 0.326 | false | 165 | 164 |
| `drift-MS_Newton_5_ENE-train-0387` | 46 | 0.279 | 0.279 | 0.279 | false | 46 | 45 |
| `drift-MS_Newton_5_ENE-train-0388` | 32 | 0.068 | 0.068 | 0.068 | false | 32 | 31 |
| `drift-NY_Ithaca_13_E-train-0447` | 238 | 0.393 | 0.393 | 0.393 | false | 238 | 236 |
| `drift-OK_Goodwell_2_E-train-0497` | 87 | 0.085 | 0.085 | 0.085 | false | 87 | 86 |
| `drift-OK_Goodwell_2_E-train-0498` | 31 | 0.206 | 0.206 | 0.206 | false | 31 | 30 |
| `drift-OK_Goodwell_2_E-train-0499` | 62 | 0.082 | 0.082 | 0.082 | false | 62 | 61 |
| `drift-OK_Goodwell_2_E-train-0500` | 27 | 0.147 | 0.147 | 0.147 | false | 27 | 26 |
| `drift-OK_Goodwell_2_SE-train-0554` | 167 | 0.109 | 0.109 | 0.109 | false | 167 | 166 |
| `drift-OK_Goodwell_2_SE-train-0555` | 71 | 0.164 | 0.164 | 0.164 | false | 71 | 70 |
| `drift-AZ_Tucson_11_W-validation-0592` | 70 | 0.036 | 0.036 | 0.036 | false | 70 | 69 |
| `drift-AZ_Tucson_11_W-validation-0593` | 49 | 0.046 | 0.046 | 0.046 | false | 49 | 48 |
| `drift-CA_Fallbrook_5_NE-validation-0628` | 107 | 0.242 | 0.242 | 0.242 | false | 107 | 106 |
| `drift-CA_Fallbrook_5_NE-validation-0629` | 24 | 0.143 | 0.143 | 0.143 | false | 24 | 23 |
| `drift-MS_Newton_5_ENE-validation-0658` | 119 | 0.333 | 0.333 | 0.333 | false | 119 | 118 |
| `drift-NY_Ithaca_13_E-validation-0694` | 37 | 0.325 | 0.325 | 0.325 | false | 37 | 36 |
| `drift-NY_Ithaca_13_E-validation-0695` | 36 | 0.268 | 0.268 | 0.268 | false | 36 | 35 |
| `drift-NY_Ithaca_13_E-validation-0696` | 25 | 0.262 | 0.262 | 0.262 | false | 25 | 24 |
| `drift-NY_Ithaca_13_E-validation-0697` | 25 | 0.356 | 0.356 | 0.356 | false | 25 | 24 |
| `drift-OK_Goodwell_2_E-validation-0725` | 69 | 0.222 | 0.222 | 0.222 | false | 69 | 68 |
| `drift-OK_Goodwell_2_E-validation-0726` | 41 | 0.169 | 0.169 | 0.169 | false | 41 | 40 |
| `drift-OK_Goodwell_2_SE-validation-0762` | 40 | 0.298 | 0.298 | 0.298 | false | 40 | 39 |
| `drift-OK_Goodwell_2_SE-validation-0763` | 78 | 0.297 | 0.297 | 0.297 | false | 78 | 77 |
| **Total : 42 fenêtres** | **2882** | | | | **tous false** | **2882** | **2839** |

## Budget et capacité Spike

Les compteurs suivants distinguent les positions réservées des modifications effectives.
Le plafond est une **borne structurelle supérieure**, avant rejets, écrêtage et
quantification. Il ne promet pas une faisabilité physique pour toutes les données.

| Famille | Planifiées | Appliquées | Rejetées | Ignorées explicitement | Quota | Réservé | Plafond | Committé | Non réservé | Réservations inutilisées | Quota non satisfait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spike | 534 | 534 | 0 | 0 | 1845 | 1845 | 1324 | 936 | 0 | 909 | 909 |
| bias | 70 | 61 | 9 | 0 | 3075 | 3051 | 3051 | 2798 | 24 | 253 | 277 |
| drift | 42 | 42 | 0 | 0 | 3074 | 2882 | 2840 | 2839 | 192 | 43 | 235 |
| replay | 75 | 75 | 0 | 0 | 2460 | 2427 | 2427 | 2416 | 33 | 11 | 44 |
| stealth | 49 | 0 | 0 | 49 | 1845 | 1767 | 0 | 0 | 78 | 1767 | 1845 |
| **Total** | **770** | **712** | **9** | **49** | **12299** | **11972** | **9642** | **8989** | **327** | **2983** | **3310** |

Identités vérifiées :

- `12299 = 11972 + 327` : allocation = réservations initiales + non-réservé.
- `11972 = 8989 + 2983` : réservations initiales = commits + libérations.
- `3310 = 327 + 2983 = 12299 - 8989` : déficit de modifications total.
- Après exécution, aucune réservation active ; le ledger conserve
  `total_reserved == total_committed == 8989`.
- Chaque réservation appliquée porte exactement le compte du masque vérifié.
  Les événements rejetés et les 49 fenêtres Stealth sont libérés.

**Spike :** 534 événements ne peuvent pas atteindre le quota de 1 845 points avec
au plus trois impacts chacun : même `534 * 3 = 1602 < 1845`.
En tenant compte des fenêtres de moins de trois positions, la vraie borne structurelle
est `sum(min(window.length, 3)) = 1324`. Le déficit structurel est donc 521 ; le tirage
du nombre d'impacts et les effets numériques/physiques laissent 388 modifications
supplémentaires non réalisées, pour 909 réservations Spike inutilisées au total.

**Amélioration minimale compatible recommandée :** définir seulement, dans une
configuration d'expérience explicite, les bornes de fenêtre Spike
`min_window: 1`, `max_window: 1`. Le scheduler existant planifie alors un événement
par observation ciblée, et le budget garde son sens de modifications. Un essai
séparé sur le dataset réel, mêmes graines et autres paramètres, a donné
**1 845 événements, 1 845 réservations, plafond 1 845, 1 845 modifications**.
Cette variante est aussi testée sur fixture. Les YAML de référence n'ont pas été changés.
Une planification distinguant quota d'impacts et capacité de fenêtre serait une évolution
ultérieure plus générale, hors de ce correctif.

## Familles exécutables

Les quatre familles exécutables restent `spike`, `bias`, `drift`, `replay`.
Stealth demeure uniquement planifiable en Phase 2.

- Politique `skip`, valeur par défaut conservée : libération et signalement explicite
  des familles et fenêtres ignorées dans la sortie texte/JSON. Dans l'expérience de
  référence : 49 fenêtres Stealth, 1 767 réservations libérées, quota de 1 845 non satisfait.
- Politique `error` : échec avant tout commit ou annulation de réservation ; tests de
  non-mutation du ledger. Usage : `--on-unsupported-family error`.
- Aucun quota n'est transféré silencieusement à une autre famille.
- Aucun code, paramètre de transformation ni modèle Stealth n'a été créé.

## Validation et reproductibilité

**308 tests passent**, sans test réel ignoré. `ruff check .` et
`ruff format --check .` passent ; 51 fichiers Python sont correctement formatés.

Le premier lancement de Pytest a rencontré un refus d'accès Windows au dossier
`%TEMP%/pytest-of-medma`. Un répertoire temporaire dédié, unique, a permis d'exécuter
la suite entière sans modifier les permissions ni la configuration du projet.

Commande PowerShell utilisée, depuis `attack-engine/` :

```powershell
$env:ATTACK_ENGINE_REAL_DATASET = (Resolve-Path ../USCRN_livraison/data/uscrn_clean.parquet).Path
$taskTestTemp = Join-Path $env:TEMP ('attack-engine-audit-' + [guid]::NewGuid().ToString('N'))
uv run --locked pytest -q --tb=short --basetemp "$taskTestTemp"
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Génération de référence, aucun fichier de données écrit :

```powershell
uv run --locked attack-engine generate --input ../USCRN_livraison/data/uscrn_clean.parquet --config configs/dataset.yaml --attack-config configs/scheduling.yaml --attack-params configs/attacks.yaml --json
```

Deux plans frais et deux exécutions avec les mêmes entrées/configurations ont donné
une égalité complète des fenêtres et des `ExecutionSummary` : valeurs, masques,
diagnostics, sources Replay et compteurs. Des assertions indépendantes ont vérifié :

- les 23 colonnes propres identiques par `assert_frame_equal(check_exact=True)` ;
- le hash du Parquet source identique avant/après ;
- chaque masque égal à la différence finie exacte propre/final ;
- toutes les candidates finies dans les bornes et aucun changement de masque NaN ;
- **8 989 lignes modifiées uniques**, sans modification chevauchante ;
- la reconstruction de la cible observée en mémoire et le ledger tous deux à 8 989 ;
- la réconciliation des cellules, des événements et des totaux.

Empreintes de comparaison des objets sérialisés en JSON trié compact
(dates en ISO 8601 ; sorties en mémoire, sans prétention de hash de fichier exporté) :

| Objet | SHA-256 |
|---|---|
| Liste des fenêtres, `model_dump(mode="json")` | `b17ce2fbc438215125d060a7f08ed647a9b2897bb71d3eabddd7ddb99c26cb19` |
| `asdict(ExecutionSummary)` | `73c6a2a767092ff47ef3b1759f68b495ce35b10e06b1146b1fc10a07c4aa5c4c` |

Ces empreintes caractérisent l'environnement et les configurations ci-dessus.
La graine seule ne garantit pas l'identité entre versions de dépendances.
La reproductibilité constatée porte sur la Phase 3 en mémoire ; aucun format d'export
ni résultat de détection n'est revendiqué.

