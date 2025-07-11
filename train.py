from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from utils import charger_donnees_json

data_path = "data/Nestle-HR-FAQ.json"
model_path = "model/sentence_bert_model.pkl"
index_path = "model/response_index.pkl"
vector_path = "model/vectorizer.pkl"

def entrainer_modele():
    print("🔄 Entraînement du chatbot RH...")

    questions, reponses = charger_donnees_json(data_path)

    print("📦 Chargement du modèle Sentence-BERT")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # léger et rapide

    print("🧠 Encodage des questions...")
    vectors = model.encode(questions)

    joblib.dump(model, model_path)
    joblib.dump(reponses, index_path)
    joblib.dump(vectors, vector_path)

    print("✅ Entraînement terminé et sauvegardé.")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    entrainer_modele()
