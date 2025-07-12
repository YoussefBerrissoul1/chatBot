import json
import os
import sys
from typing import Tuple, Dict, List
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from spellchecker import preprocess_question
from utils import charger_donnees_json
from logger import QuestionLogger

class ChatbotRH:
    """Chatbot RH Nestlé utilisant BERT et FAISS pour répondre aux questions RH."""
    
    def __init__(self,
                 model_path: str = "model/sentence_bert_model.pkl",
                 intent_model_path: str = "model/intent_classifier.pkl",
                 data_path: str = "data/Nestle-HR-FAQ.json",
                 log_path: str = "log/incompris.json",
                 seuil_confiance_haute: float = 0.75,
                 seuil_confiance_basse: float = 0.5):
        """Initialise le chatbot avec les paramètres de configuration."""
        self.model_path = model_path
        self.intent_model_path = intent_model_path
        self.data_path = data_path
        self.log_path = log_path
        self.seuil_confiance_haute = seuil_confiance_haute
        self.seuil_confiance_basse = seuil_confiance_basse
        self.model = None
        self.intent_model = None
        self.logger = None
        self.theme_indexes: Dict[str, Tuple[List[str], List[str], faiss.Index]] = {}
        self.full_data: Dict = {}
        self.stats = {
            "questions_posees": 0,
            "reponses_directes": 0,
            "suggestions": 0,
            "incompris": 0
        }

    def charger_modeles(self) -> None:
        """Charge les modèles SentenceTransformer et de classification d'intention."""
        try:
            print("📦 Chargement des modèles...")
            self.model = joblib.load(self.model_path)
            self.intent_model = joblib.load(self.intent_model_path)
            self.logger = QuestionLogger(self.log_path)
            print("✅ Modèles chargés avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles : {e}")
            sys.exit(1)

    def charger_donnees_faq(self) -> None:
        """Charge les données FAQ à partir du fichier JSON."""
        try:
            print("📂 Chargement des données FAQ...")
            self.full_data = charger_donnees_json(self.data_path)
            print(f"✅ {len(self.full_data)} thèmes RH chargés")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données FAQ : {e}")
            sys.exit(1)

    def creer_index_faiss(self) -> None:
        """Crée les index FAISS pour chaque thème FAQ."""
        try:
            print("📌 Création des index FAISS...")
            for theme, items in self.full_data.items():
                questions = [item["question"] for item in items]
                reponses = [item["response"] for item in items]
                vectors = self.model.encode(questions, convert_to_numpy=True).astype("float32")
                index = faiss.IndexFlatL2(vectors.shape[1])
                index.add(vectors)
                self.theme_indexes[theme] = (questions, reponses, index)
            print("✅ Index FAISS créés")
        except Exception as e:
            print(f"❌ Erreur lors de la création des index FAISS : {e}")
            sys.exit(1)

    def detecter_intention(self, question: str) -> str:
        """Détecte l'intention de la question."""
        try:
            return self.intent_model.predict([question])[0]
        except Exception as e:
            print(f"⚠️ Erreur de détection d'intention : {e}")
            return list(self.full_data.keys())[0]

    def rechercher_reponse(self, question: str, theme: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """Recherche les réponses pertinentes pour une question donnée."""
        questions, reponses, index = self.theme_indexes.get(theme, ([], [], None))
        if not questions or not index:
            return [], []
        
        vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
        D, I = index.search(vec, k=min(k, len(questions)))
        scores = 1 - D[0] / 2  # Convertir L2 en score [0, 1]
        return [reponses[i] for i in I[0]], scores.tolist()

    def generer_reponse(self, question: str) -> str:
        """Génère une réponse à partir de la question posée."""
        self.stats["questions_posees"] += 1
        question_corrigee = preprocess_question(question)
        if len(question_corrigee.split()) < 3:
            question_corrigee = f"je veux savoir {question_corrigee}"

        theme = self.detecter_intention(question_corrigee)
        print(f"📂 Section détectée : {theme.replace('_', ' ').title()}")
        
        reponses, scores = self.rechercher_reponse(question_corrigee, theme)
        if not reponses:
            self.stats["incompris"] += 1
            self.logger.enregistrer_question(question, section_detectee=theme)
            return "❌ Aucune réponse trouvée pour ce thème."

        best_score = scores[0]
        best_reponse = reponses[0]

        if best_score >= self.seuil_confiance_haute:
            self.stats["reponses_directes"] += 1
            return f"🎯 {best_reponse}"
        elif best_score >= self.seuil_confiance_basse:
            self.stats["suggestions"] += 1
            suggestions = [r for r, s in zip(reponses, scores) if s >= self.seuil_confiance_basse][:2]
            return "🤔 Voici quelques suggestions :\n" + "\n".join(f"  - {s}" for s in suggestions)
        else:
            self.stats["incompris"] += 1
            self.logger.enregistrer_question(question, section_detectee=theme)
            return "❌ Je n'ai pas compris votre question. Reformulez-la SVP."

    def initialiser(self) -> None:
        """Initialise le chatbot en chargeant modèles, données et index."""
        print("🚀 Initialisation du Chatbot RH Nestlé...")
        self.charger_modeles()
        self.charger_donnees_faq()
        self.creer_index_faiss()
        print("✅ Chatbot prêt à répondre !")

    def afficher_statistiques(self) -> None:
        """Affiche les statistiques de la session."""
        print("\n📊 STATISTIQUES DE SESSION :")
        for key, value in self.stats.items():
            print(f"  {key.replace('_', ' ').capitalize()} : {value}")
        if self.stats["questions_posees"] > 0:
            taux = ((self.stats["reponses_directes"] + self.stats["suggestions"]) /
                    self.stats["questions_posees"]) * 100
            print(f"  Taux de compréhension : {taux:.1f}%")

    def executer(self) -> None:
        """Exécute la boucle principale du chatbot."""
        print("💬 Bienvenue sur le chatbot RH Nestlé (exit pour quitter)\n")
        while True:
            try:
                question = input("Vous: ").strip()
                if question.lower() in ["exit", "quit"]:
                    print("👋 Merci d’avoir utilisé le chatbot RH.")
                    self.afficher_statistiques()
                    break
                if not question:
                    print("⚠️ Veuillez poser une question.")
                    continue

                reponse = self.generer_reponse(question)
                print(f"Bot: {reponse}\n")

            except KeyboardInterrupt:
                print("\n👋 Interruption manuelle.")
                self.afficher_statistiques()
                break
            except Exception as e:
                print(f"❌ Erreur inattendue : {e}")
                continue

def main() -> None:
    """Point d'entrée principal du programme."""
    chatbot = ChatbotRH()
    chatbot.initialiser()
    chatbot.executer()

if __name__ == "__main__":
    main()