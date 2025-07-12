import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import charger_donnees_json
import os

model_output = "model/intent_classifier.pkl"

def preparer_data_par_theme(chemin):
    import json
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions, labels = [], []
    for theme, items in data["faq"].items():
        for item in items:
            questions.append(item["question"])
            labels.append(theme)
    return questions, labels

def entrainer_classifieur_intention():
    print("🎯 Entraînement du classifieur d’intention RH...")

    questions, themes = preparer_data_par_theme("data/Nestle-HR-FAQ.json")

    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    model.fit(questions, themes)

    joblib.dump(model, model_output)
    print(f"✅ Modèle de classification sauvé → {model_output}")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    entrainer_classifieur_intention()
