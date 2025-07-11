import json

def charger_donnees_json(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions, reponses = [], []
    for theme in data["faq"].values():
        for item in theme:
            questions.append(item["question"])
            reponses.append(item["response"])
    return questions, reponses
