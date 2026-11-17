# %% [markdown]
# # Comparateur de modèles d'IA frugale : précision / latence / énergie
# Tâche : analyse de sentiments (SST-2). Mesures avec CodeCarbon, inférence sur CPU.
#
# Installation (terminal VS Code) :
#   pip install codecarbon scikit-learn datasets transformers torch pandas matplotlib
#
# Utilisation : ouvrir dans VS Code, exécuter cellule par cellule ("Run Cell").
# ATTENTION : ce script n'a pas été exécuté par son auteur (pas d'accès réseau).
# Vérifiez les noms de modèles Hugging Face (section 2) et corrigez si besoin.

# %% 1. Paramètres
import time, json, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

SEED = 42
N_REPEATS = 3          # répétitions par modèle (pour moyenne et écart-type)
MIN_SECONDS = 20       # durée minimale de chaque mesure (CodeCarbon échantillonne ~1/s)
BATCH_SIZE = 32
COUNTRY = "FRA"        # pays utilisé par CodeCarbon pour l'intensité carbone

# Intensités carbone (gCO2e/kWh, cycle de vie, données 2025) pour le scénario "autre pays".
# Source : Ember (2026), Yearly Electricity Data, via Our World in Data
# (ourworldindata.org/electricity-mix, licence CC BY).
INTENSITES = {"France": 41.44, "Allemagne": 329.65, "Pologne": 588.6}

random.seed(SEED); np.random.seed(SEED)

# %% 2. Données et modèles
ds = load_dataset("nyu-mll/glue", "sst2")
train_small = ds["train"].shuffle(seed=SEED).select(range(10000))   # pour TF-IDF
test = ds["validation"]                                              # 872 exemples
X_test, y_test = list(test["sentence"]), list(test["label"])

# Modèles pré-entraînés sur SST-2 (noms à vérifier sur huggingface.co)
HF_MODELS = {
    "TinyBERT (distillé)": "philschmid/tiny-bert-sst2-distilled",
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "BERT-base": "textattack/bert-base-uncased-SST-2",
}

# %% 3. Fonction de mesure générique
def nouveau_tracker():
    """Crée un EmissionsTracker en s'adaptant aux différentes versions de CodeCarbon."""
    essais = [
        dict(measure_power_secs=1, save_to_file=False, log_level="error", country_iso_code=COUNTRY),
        dict(measure_power_secs=1, save_to_file=False, log_level="error"),
        dict(measure_power_secs=1, save_to_file=False),
        dict(),
    ]
    for kw in essais:
        try:
            return EmissionsTracker(**kw)
        except TypeError:
            continue
    raise RuntimeError("Impossible de créer l'EmissionsTracker : envoyez la version de codecarbon.")

def mesurer(nom, predict_fn, n_repeats=N_REPEATS):
    """predict_fn(textes) -> liste de labels. Boucle sur le jeu de test jusqu'à MIN_SECONDS."""
    lignes = []
    for r in range(n_repeats):
        tracker = nouveau_tracker()
        n_pred, preds = 0, None
        tracker.start()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < MIN_SECONDS:
            preds = predict_fn(X_test)
            n_pred += len(X_test)
        duree = time.perf_counter() - t0
        kg = tracker.stop()
        kwh = tracker.final_emissions_data.energy_consumed
        lignes.append({
            "modele": nom, "run": r,
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "latence_ms": 1000 * duree / n_pred,
            "Wh_par_1000": 1000 * kwh * 1000 / n_pred,   # kWh -> Wh, puis /1000 prédictions
            "gCO2e_par_1000": 1000 * kg * 1000 / n_pred, # kg -> g
            # Infos utiles pour discuter la précision de la mesure (limite sous Windows)
            "cpu_power_W": getattr(tracker.final_emissions_data, "cpu_power", None),
            "ram_power_W": getattr(tracker.final_emissions_data, "ram_power", None),
            "gpu_power_W": getattr(tracker.final_emissions_data, "gpu_power", None),
            "tracking_mode": getattr(tracker.final_emissions_data, "tracking_mode", None),
        })
        print(nom, r, {k: round(v, 4) for k, v in lignes[-1].items()
                       if isinstance(v, (int, float)) and k != "run"})
    return lignes

resultats = []

# %% 4. Modèle 1 : TF-IDF + régression logistique (entraînement mesuré à part)
tr = nouveau_tracker()
tr.start()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(train_small["sentence"])
clf = LogisticRegression(max_iter=1000).fit(Xtr, train_small["label"])
tr.stop()
energie_entrainement_tfidf_Wh = tr.final_emissions_data.energy_consumed * 1000
print("Énergie d'entraînement TF-IDF (Wh):", energie_entrainement_tfidf_Wh)

resultats += mesurer("TF-IDF + LogReg", lambda t: clf.predict(vec.transform(t)))

# %% 5. Modèles Transformers (inférence CPU uniquement)
import torch
from transformers import pipeline
torch.set_num_threads(4)   # fixer pour la reproductibilité

for nom, repo in HF_MODELS.items():
    pipe = pipeline("text-classification", model=repo, device=-1, truncation=True)
    def predict(textes, pipe=pipe):
        sorties = pipe(textes, batch_size=BATCH_SIZE)
        return [1 if s["label"].upper() in ("POSITIVE", "LABEL_1") else 0 for s in sorties]
    predict(X_test[:BATCH_SIZE])    # échauffement (non mesuré)
    resultats += mesurer(nom, predict)
    del pipe

# %% 6. Agrégation, scénarios pays, export
df = pd.DataFrame(resultats)
df.to_csv("resultats_bruts.csv", index=False)
print("Mode de suivi CodeCarbon :", df["tracking_mode"].unique())
num = df.drop(columns=["tracking_mode"]).apply(
    lambda c: pd.to_numeric(c, errors="coerce") if c.name != "modele" else c)
agg = num.groupby("modele").agg(["mean", "std"]).drop(columns="run")
agg.to_csv("resultats_agreges.csv")
print(agg.round(4))

moy = num.groupby("modele").mean(numeric_only=True)
scenarios = pd.DataFrame({pays: moy["Wh_par_1000"] / 1000 * g   # kWh * gCO2/kWh
                          for pays, g in INTENSITES.items()})
scenarios.columns = [f"gCO2e/1000 pred. ({p})" for p in scenarios.columns]
scenarios.to_csv("scenarios_pays.csv")
print(scenarios.round(4))

# %% 7. Graphiques
fig, ax = plt.subplots(figsize=(7, 5))
for nom, ligne in moy.iterrows():
    ax.scatter(ligne["Wh_par_1000"], ligne["accuracy"] * 100, s=120)
    ax.annotate(nom, (ligne["Wh_par_1000"], ligne["accuracy"] * 100),
                textcoords="offset points", xytext=(6, 6))
ax.set_xscale("log")
ax.set_xlabel("Énergie (Wh pour 1000 prédictions, échelle log)")
ax.set_ylabel("Précision (%)")
ax.set_title("Compromis précision / énergie (SST-2, CPU)")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig("fig_precision_energie.png", dpi=300)

fig2, ax2 = plt.subplots(figsize=(7, 4))
scenarios.plot(kind="bar", ax=ax2)
ax2.set_ylabel("gCO2e pour 1000 prédictions"); ax2.set_xlabel("")
ax2.set_title("Effet du pays (intensité carbone)")
plt.xticks(rotation=20); fig2.tight_layout()
fig2.savefig("fig_scenarios_pays.png", dpi=300)

# %% 8. Informations machine à citer dans le poster (reproductibilité)
import platform
info = {"os": platform.platform(), "cpu": platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "n_test": len(X_test),
        "min_seconds": MIN_SECONDS, "repeats": N_REPEATS}
json.dump(info, open("infos_machine.json", "w"), indent=2)
print(info)
