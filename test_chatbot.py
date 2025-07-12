"""
test_chatbot.py
---------------
Script de test automatique du chatbot RH Nestlé
"""

from chatbot import ChatbotRH
import json

# Exemple de questions à tester (tu peux les remplacer par celles de ton domaine RH)
questions_test = [
    "Comment puis-je poser mes congés ?",
    "Quels sont les avantages d'un contrat CDI ?",
    "Est-ce que je peux avoir un bulletin de paie imprimé ?",
    "Que faire en cas de retard de paie ?",
    "Combien de jours de télétravail sont autorisés ?",
    "Nestlé propose-t-il une mutuelle santé ?",
    "Mon salaire est-il versé à la fin du mois ?",
    "Puis-je faire une demande d’avancement ?",
    "Y a-t-il une politique de travail hybride ?",
    "Comment faire si je suis victime de harcèlement ?"
]

def tester_chatbot(questions):
    chatbot = ChatbotRH()
    chatbot.initialiser()

    total = len(questions)
    bien_comprises = 0

    print("\n📊 Début des tests...\n")
    for i, q in enumerate(questions, 1):
        print(f"🔹 Q{i}: {q}")
        reponse = chatbot.generer_reponse(q)
        print(f"🔸 Réponse: {reponse}\n")

        # Estimation : si pas "❌", on considère la question comprise
        if "❌" not in reponse:
            bien_comprises += 1

    print("🧾 Résumé :")
    print(f"✅ Questions comprises : {bien_comprises}/{total}")
    print(f"📈 Taux de compréhension : {(bien_comprises/total)*100:.1f}%\n")

if __name__ == "__main__":
    tester_chatbot(questions_test)
