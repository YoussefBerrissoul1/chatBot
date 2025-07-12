from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import faiss
from utils import charger_donnees_json

# 📦 Chargement des modèles
model = joblib.load("model/sentence_bert_model.pkl")
intent_model = joblib.load("model/intent_classifier.pkl")

# 📂 Chargement complet des données
all_questions, all_reponses = charger_donnees_json("data/Nestle-HR-FAQ.json")

# 🔀 Groupement par thème (nécessaire pour filtrer par catégorie)
import json
with open("data/Nestle-HR-FAQ.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)["faq"]

# 📌 Création des vecteurs groupés par thème
# (dictionnaire : {theme: (questions, réponses, index FAISS)})
theme_indexes = {}

for theme, qa_list in full_data.items():
    questions = [item["question"] for item in qa_list]
    reponses = [item["response"] for item in qa_list]
    vectors = model.encode(questions, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    theme_indexes[theme] = (questions, reponses, index)

# 🧠 Fonction principale
def chatbot():
    print("💬 Chatbot RH Nestlé (tapez 'exit' pour quitter)")
    while True:
        question = input("Vous: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Merci, à bientôt !")
            break

        # 🎯 Prédiction du thème
        theme = intent_model.predict([question])[0]
        print(f"📂 Section détectée : {theme.replace('_', ' ').title()}")

        questions, reponses, index = theme_indexes[theme]

        vec = model.encode([question], convert_to_numpy=True).astype("float32")
        D, I = index.search(vec, k=1)
        score = 1 - D[0][0] / 2  # approx. [0, 1]
        idx = I[0][0]

        if score > 0.75:
            print("Bot:", reponses[idx])
        elif score > 0.5:
            print("Bot: 🤔 Peut-être vouliez-vous dire :")
            print("   ", reponses[idx])
        else:
            print("Bot: ❌ Je n’ai pas compris. Reformulez svp.")

if __name__ == "__main__":
    chatbot()
