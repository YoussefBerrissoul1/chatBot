import json
import os
from datetime import datetime

class QuestionLogger:
    """
    Classe pour enregistrer les questions mal comprises par le chatbot RH
    dans un fichier JSON avec horodatage.
    """

    def __init__(self, log_path: str = "logs/unanswered_questions.json"):
        """
        Initialise le logger et crée le fichier si nécessaire.
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Initialiser le fichier si vide ou inexistant
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump({"questions": []}, f, indent=2, ensure_ascii=False)

    def enregistrer_question(self, question: str, section_detectee: str = None):
        """
        Ajoute une entrée au fichier log avec la question et le timestamp.
        
        Args:
            question (str): La question saisie par l'utilisateur
            section_detectee (str, optional): La section RH détectée (si applicable)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Charger le fichier actuel
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"questions": []}

        # Ajouter la nouvelle entrée
        data["questions"].append({
            "question": question,
            "section": section_detectee,
            "timestamp": timestamp
        })

        # Sauvegarder le fichier mis à jour
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"📝 Question enregistrée dans le log : {self.log_path}")
