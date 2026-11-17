# IA frugale : quel compromis entre précision et énergie ?

Projet d'informatique soutenable (Université d'Évry – Paris-Saclay), novembre 2026.
Auteures : Nada Hamadache, Fatima Ouhocine, Mohamed Fazul Raissa. Encadrant : Vincent Chau.

## Objectif

Comparer, pour une même tâche (analyse de sentiments sur SST-2), quatre modèles
du plus sobre au plus lourd et tracer le compromis **précision / latence / énergie**.
Le projet s'appuie sur l'article de Henderson et al. (2020), *Towards the Systematic
Reporting of the Energy and Carbon Footprints of Machine Learning* (JMLR 21, 1-43),
dont il reprend l'idée de rapporter systématiquement énergie et carbone, avec un outil
différent (CodeCarbon) et sur une machine Windows.

## Modèles comparés

| Modèle | Rôle |
|---|---|
| TF-IDF + régression logistique | Modèle le plus sobre (entraîné sur 10 000 phrases de SST-2) |
| TinyBERT (distillé) | Petit modèle pré-entraîné sur SST-2 |
| DistilBERT | Modèle intermédiaire |
| BERT-base | Situation de référence (modèle le plus lourd) |

Les modèles pré-entraînés sont téléchargés depuis Hugging Face (identifiants dans la
section 2 du script). Les noms ont été vérifiés lors de l'exécution du script, mais les
dépôts distants peuvent évoluer.

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `benchmark_ia_frugale.py` | Script complet (cellules `# %%` utilisables dans VS Code) |
| `resultats_bruts.csv` | Une ligne par modèle et par répétition |
| `resultats_agreges.csv` | Moyenne et écart-type sur 3 répétitions |
| `scenarios_pays.csv` | Émissions pour 1000 prédictions selon le pays (voir ci-dessous) |
| `infos_machine.json` | Système, versions et paramètres de l'exécution |
| `fig_poster_precision_energie.png`, `fig_poster_pays.png` | Figures du poster |
| `requirements.txt` | Versions des bibliothèques utilisées |

## Reproduire les résultats

1. Installer Python et créer un environnement :
   ```
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   ```
   (Sous Linux ou macOS : `source env/bin/activate`.)
2. Placer `benchmark_ia_frugale.py` dans un dossier **hors de OneDrive** (la synchronisation
   ralentit fortement le chargement de `torch`).
3. Brancher la machine sur secteur, fermer les autres programmes, puis lancer :
   ```
   python benchmark_ia_frugale.py
   ```
   Compter 20 à 40 minutes (connexion Internet requise au premier lancement).

Le script produit les fichiers CSV, les deux figures et `infos_machine.json`.

## Conditions de mesure

- Système : Windows 11, processeur Intel (famille 6, modèle 142), CPU uniquement, 4 threads.
- Python 3.14.3, CodeCarbon 3.3.2 (autres versions dans `requirements.txt` et `infos_machine.json`).
- Jeu de test : 872 phrases (validation SST-2). Chaque mesure dure au moins 20 s ; 3 répétitions par modèle.
- Énergie en Wh pour 1000 prédictions, rapportée par CodeCarbon.

## Résultats (moyenne de 3 répétitions)

| Modèle | Précision (%) | Latence (ms/phrase) | Énergie (Wh / 1000 pred.) |
|---|---|---|---|
| TF-IDF + LogReg | 78,6 | 0,022 | 0,00015 |
| TinyBERT | 83,4 | 0,81 | 0,0065 |
| DistilBERT | 91,1 | 24,5 | 0,176 |
| BERT-base | 92,4 | 52,5 | 0,356 |

L'écart-type de l'énergie entre les répétitions est de 2 à 5 %.

## Scénarios par pays

`scenarios_pays.csv` multiplie l'énergie mesurée par l'intensité carbone du réseau
électrique, en gCO₂e/kWh, cycle de vie, données 2025 :
France 41,44 ; Allemagne 329,65 ; Pologne 588,6.
Source : Ember (2026), *Yearly Electricity Data*, via Our World in Data
(ourworldindata.org/electricity-mix, licence CC BY).

Le fichier fourni a été recalculé à partir des mêmes mesures avec ces intensités
(le script, relancé, refait les mesures et régénère le fichier avec les mêmes valeurs).
D'autres sources donnent des valeurs différentes selon la méthode (cycle de vie ou
émissions directes, production ou consommation).

## Limites

- Sous Windows, l'énergie est **estimée** par CodeCarbon et non mesurée directement par
  les compteurs matériels. Seules les comparaisons relatives entre modèles, faites dans
  les mêmes conditions, sont fiables ; les valeurs absolues sont approximatives.
- Une seule tâche, une seule machine, 3 répétitions.
- Le modèle TF-IDF n'est entraîné que sur 10 000 phrases, et son énergie, très faible,
  est proche de la limite de précision de la mesure.
- L'écart de précision entre DistilBERT et BERT-base (1,4 point) est proche de
  l'incertitude statistique d'un test de 872 phrases.
- La fabrication du matériel n'est pas comptée (comme dans l'article de référence).

## Références

- Henderson, P. et al. (2020). *Towards the Systematic Reporting of the Energy and Carbon
  Footprints of Machine Learning*. JMLR 21, 1-43.
- CodeCarbon : https://codecarbon.io
- SST-2 (Socher et al., 2013), via GLUE (Wang et al., 2019), jeu `nyu-mll/glue` sur Hugging Face.
- Ember (2026), Yearly Electricity Data, via Our World in Data.
