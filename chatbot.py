# 📄 chatbot.py
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np
import faiss

model = joblib.load("model/sentence_bert_model.pkl")
reponses = joblib.load("model/response_index.pkl")
index = faiss.read_index("model/faiss.index")

def chatbot():
    print("\U0001F4AC Chatbot RH Nestl\u00e9 (tapez 'exit' pour quitter)")
    while True:
        question = input("Vous: ")
        if question.lower() in ["exit", "quit"]:
            print("\U0001F44B Merci, \u00e0 bient\u00f4t !")
            break
        vec = model.encode([question], convert_to_numpy=True)
        D, I = index.search(np.array(vec).astype("float32"), k=1)
        score = 1 - D[0][0] / 2  # approx sim to [0, 1]
        idx = I[0][0]

        if score > 0.7:
            print("Bot:", reponses[idx])
        elif score > 0.5:
            print("Bot: Peut-\u00eatre vouliez-vous dire :", reponses[idx])
        else:
            print("Bot: \u274c Je n\u2019ai pas compris. Reformulez svp.")


if __name__ == "__main__":
    chatbot()
