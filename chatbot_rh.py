# import json
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Charger les questions/réponses
# def charger_faq(chemin_fichier):
#     with open(chemin_fichier, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     questions, reponses = [], []
#     for theme in data["faq"].values():
#         for item in theme:
#             questions.append(item["question"])
#             reponses.append(item["response"])
#     return questions, reponses

# # Trouver la réponse à une question posée
# def trouver_reponse(question_utilisateur, questions, reponses):
#     vecteur = TfidfVectorizer()
#     tfidf_matrix = vecteur.fit_transform(questions + [question_utilisateur])
#     similarites = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
#     index_max = similarites.argmax()
#     score_max = similarites[0, index_max]
#     if score_max > 0.6:
#         return reponses[index_max]
#     else:
#         return "Je suis désolé, je n’ai pas compris votre question. Veuillez reformuler."

# # Fonction principale
# def main():
#     chemin = "Nestle-HR-FAQ.json"  # Mets le chemin si nécessaire
#     questions, reponses = charger_faq(chemin)
#     print("💬 Chatbot RH Nestlé (tapez 'exit' pour quitter)")
#     while True:
#         question = input("Vous: ")
#         if question.lower() in ['exit', 'quit']:
#             print("👋 Merci, à bientôt !")
#             break
#         reponse = trouver_reponse(question, questions, reponses)
#         print("Bot:", reponse)

# if __name__ == "__main__":
#     main()
