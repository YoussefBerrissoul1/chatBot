from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np

model = joblib.load("model/sentence_bert_model.pkl")
reponses = joblib.load("model/response_index.pkl")
vectors = joblib.load("model/vectorizer.pkl")

def chatbot():
    print("💬 Chatbot RH Nestlé (tapez 'exit' pour quitter)")
    while True:
        question = input("Vous: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Merci, à bientôt !")
            break
        vec_question = model.encode([question])
        sims = cosine_similarity(vec_question, vectors)[0]
        idx = np.argmax(sims)
        score = sims[idx]

        if score > 0.7:
            print("Bot:", reponses[idx])
        elif score > 0.4:
            print("Bot: Vous voulez dire :", reponses[idx])
        else:
            print("Bot: ❌ Je n’ai pas compris. Reformulez svp.")

if __name__ == "__main__":
    chatbot()
