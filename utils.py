import json

def charger_donnees_json(chemin: str) -> dict:
    """
    Charge les données JSON au format {"faq": {theme: [questions]}} pour le chatbot.
    Utilisé par : ChatbotRH
    """
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "faq" not in data:
        raise ValueError("❌ Clé 'faq' manquante dans le fichier JSON")

    return data["faq"]

def charger_questions_reponses(chemin: str):
    """
    Charge les questions et réponses à plat (liste) pour entraînement.
    Utilisé par : train.py, intent_classifier.py
    """
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions, reponses = [], []
    for theme in data["faq"].values():
        for item in theme:
            questions.append(item["question"])
            reponses.append(item["response"])
    return questions, reponses
