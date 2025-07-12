from sentence_transformers import SentenceTransformer
import joblib
import faiss
import numpy as np
import os
from utils import charger_donnees_json

data_path = "data/Nestle-HR-FAQ.json"
model_path = "model/sentence_bert_model.pkl"
index_path = "model/response_index.pkl"
faiss_index_path = "model/faiss.index"

def entrainer_modele():
    print("🔄 Entraînement du chatbot RH...")

    questions, reponses = charger_donnees_json(data_path)

    print("📦 Chargement du modèle Sentence-BERT amélioré")
    model = SentenceTransformer("all-mpnet-base-v2")  # Meilleur que MiniLM

    print("🧠 Encodage des questions...")
    vectors = model.encode(questions, convert_to_numpy=True)

    print("⚡ Construction de l'index FAISS...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))

    print("💾 Sauvegarde du modèle et index...")
    joblib.dump(model, model_path)
    joblib.dump(reponses, index_path)
    faiss.write_index(index, faiss_index_path)

    print("✅ Entraînement terminé et sauvegardé.")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    entrainer_modele()
