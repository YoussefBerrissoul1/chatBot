# """
# Chatbot RH Nestlé - Assistant conversationnel intelligent
# Auteur: Système de chatbot RH
# Description: Chatbot utilisant BERT et FAISS pour répondre aux questions RH avec correction orthographique.
# """

# import json
# import os
# import sys
# from typing import Tuple, List, Dict, Any
# import joblib
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from spellchecker import preprocess_question
# from utils import charger_donnees_json
# from logger import QuestionLogger


# class ChatbotRH:
#     """
#     Chatbot RH utilisant BERT et FAISS pour répondre aux questions des employés.

#     Attributes:
#         model_path (str): Chemin vers le modèle BERT
#         intent_model_path (str): Chemin vers le modèle de classification d'intentions
#         data_path (str): Chemin vers les données FAQ
#         log_path (str): Chemin vers le fichier de log
#         seuil_confiance_haute (float): Seuil pour réponses directes
#         seuil_confiance_basse (float): Seuil pour suggestions
#     """

#     def __init__(
#         self,
#         model_path: str = "model/sentence_bert_model.pkl",
#         intent_model_path: str = "model/intent_classifier.pkl",
#         data_path: str = "data/Nestle-HR-FAQ.json",
#         log_path: str = "log/incompris.json",
#         seuil_confiance_haute: float = 0.75,
#         seuil_confiance_basse: float = 0.5,
#     ):
#         """
#         Initialise le chatbot RH avec les paramètres spécifiés.

#         Args:
#             model_path: Chemin vers le modèle BERT
#             intent_model_path: Chemin vers le modèle de classification d'intentions
#             data_path: Chemin vers les données FAQ
#             log_path: Chemin vers le fichier de log
#             seuil_confiance_haute: Seuil pour réponses directes (défaut: 0.75)
#             seuil_confiance_basse: Seuil pour suggestions (défaut: 0.5)
#         """
#         self.model_path = model_path
#         self.intent_model_path = intent_model_path
#         self.data_path = data_path
#         self.log_path = log_path
#         self.seuil_confiance_haute = seuil_confiance_haute
#         self.seuil_confiance_basse = seuil_confiance_basse
#         self.model: SentenceTransformer = None
#         self.intent_model = None
#         self.logger: QuestionLogger = None
#         self.theme_indexes: Dict[str, Tuple[List[str], List[str], faiss.Index]] = {}
#         self.full_data: Dict[str, List[Dict[str, str]]] = {}
#         self.stats = {
#             "questions_posees": 0,
#             "reponses_directes": 0,
#             "suggestions": 0,
#             "incompris": 0,
#         }

#     def charger_modeles(self) -> None:
#         """
#         Charge les modèles BERT et de classification d'intentions.

#         Raises:
#             FileNotFoundError: Si les fichiers de modèles n'existent pas
#             Exception: Si le chargement échoue
#         """
#         print("📦 Chargement des modèles...")
#         try:
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"Modèle BERT non trouvé : {self.model_path}")
#             if not os.path.exists(self.intent_model_path):
#                 raise FileNotFoundError(f"Modèle d'intention non trouvé : {self.intent_model_path}")
            
#             self.model = joblib.load(self.model_path)
#             self.intent_model = joblib.load(self.intent_model_path)
#             self.logger = QuestionLogger(self.log_path)
#             print("✅ Modèles chargés avec succès")
#         except Exception as e:
#             print(f"❌ Erreur lors du chargement des modèles : {e}")
#             sys.exit(1)

#     def charger_donnees_faq(self) -> None:
#         """
#         Charge les données FAQ depuis le fichier JSON.

#         Raises:
#             FileNotFoundError: Si le fichier FAQ n'existe pas
#             json.JSONDecodeError: Si le fichier JSON est malformé
#         """
#         print("📂 Chargement des données FAQ...")
#         try:
#             if not os.path.exists(self.data_path):
#                 raise FileNotFoundError(f"Fichier FAQ non trouvé : {self.data_path}")
            
#             self.full_data = charger_donnees_json(self.data_path)
#             print(f"✅ {len(self.full_data)} thèmes RH chargés")
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             print(f"❌ Erreur lors du chargement des données FAQ : {e}")
#             sys.exit(1)

#     def creer_index_faiss(self, force: bool = False) -> None:
#         """
#         Crée ou recharge les index FAISS pour chaque thème RH et les stocke dans le dossier model/indexes/.

#         Args:
#             force: Si True, force la reconstruction des index (supprime le cache)

#         Raises:
#             Exception: Si la création/chargement des index échoue
#         """
#         dossier_index = "model/indexes"
#         os.makedirs(dossier_index, exist_ok=True)

#         if force:
#             print("♻️ Mode REBUILD : suppression des anciens index...")
#             for fname in os.listdir(dossier_index):
#                 os.remove(os.path.join(dossier_index, fname))

#         print("📌 Chargement ou création des index FAISS par thème...")
#         try:
#             for theme, qa_list in self.full_data.items():
#                 questions = [item["question"] for item in qa_list]
#                 reponses = [item["response"] for item in qa_list]
#                 index_path = os.path.join(dossier_index, f"{theme}.index")
#                 question_path = os.path.join(dossier_index, f"{theme}_questions.json")
#                 reponse_path = os.path.join(dossier_index, f"{theme}_reponses.json")

#                 if not force and all(os.path.exists(p) for p in [index_path, question_path, reponse_path]):
#                     try:
#                         index = faiss.read_index(index_path)
#                         with open(question_path, "r", encoding="utf-8") as f:
#                             questions = json.load(f)
#                         with open(reponse_path, "r", encoding="utf-8") as f:
#                             reponses = json.load(f)
#                         print(f"🔁 Index FAISS chargé depuis cache pour : {theme}")
#                     except Exception as e:
#                         print(f"⚠️ Erreur chargement index FAISS [{theme}], réencodage forcé... → {e}")
#                         self._creer_nouvel_index(theme, questions, reponses, index_path, question_path, reponse_path)
#                         index = faiss.read_index(index_path)
#                 else:
#                     print(f"➕ Création de l'index FAISS pour : {theme}")
#                     self._creer_nouvel_index(theme, questions, reponses, index_path, question_path, reponse_path)
#                     index = faiss.read_index(index_path)

#                 self.theme_indexes[theme] = (questions, reponses, index)

#             print(f"✅ {len(self.theme_indexes)} index FAISS disponibles.")
#         except Exception as e:
#             print(f"❌ Erreur lors de la création/chargement des index FAISS : {e}")
#             sys.exit(1)

#     def _creer_nouvel_index(self, theme: str, questions: List[str], reponses: List[str],
#                            index_path: str, question_path: str, reponse_path: str) -> None:
#         """
#         Crée un nouvel index FAISS et sauvegarde les fichiers de cache.

#         Args:
#             theme: Nom du thème
#             questions: Liste des questions
#             reponses: Liste des réponses
#             index_path: Chemin du fichier d'index FAISS
#             question_path: Chemin du fichier des questions
#             reponse_path: Chemin du fichier des réponses
#         """
#         try:
#             vectors = self.model.encode(questions, convert_to_numpy=True).astype("float32")
#             index = faiss.IndexFlatL2(vectors.shape[1])
#             index.add(vectors)
#             faiss.write_index(index, index_path)
#             with open(question_path, "w", encoding="utf-8") as f:
#                 json.dump(questions, f, ensure_ascii=False, indent=2)
#             with open(reponse_path, "w", encoding="utf-8") as f:
#                 json.dump(reponses, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             print(f"❌ Erreur lors de la création de l'index pour {theme} : {e}")
#             raise

#     def detecter_intention(self, question: str) -> str:
#         """
#         Détecte l'intention/thème de la question posée.

#         Args:
#             question: La question de l'utilisateur

#         Returns:
#             str: Le thème détecté
#         """
#         try:
#             return self.intent_model.predict([question])[0]
#         except Exception as e:
#             print(f"⚠️ Erreur détection intention : {e}")
#             return list(self.full_data.keys())[0]

#     def rechercher_reponse(self, question: str, theme: str, k: int = 3) -> Tuple[List[str], List[float]]:
#         """
#         Recherche les meilleures réponses pour une question dans un thème donné.

#         Args:
#             question: La question à traiter
#             theme: Le thème dans lequel rechercher
#             k: Nombre de réponses à retourner

#         Returns:
#             Tuple[List[str], List[float]]: Les réponses et leurs scores de confiance
#         """
#         if theme not in self.theme_indexes:
#             print(f"⚠️ Thème '{theme}' non trouvé")
#             return [], []

#         questions, reponses, index = self.theme_indexes[theme]
#         vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
#         D, I = index.search(vec, k=min(k, len(questions)))
#         scores = 1 - D[0] / 2
#         return [reponses[i] for i in I[0]], scores.tolist()

#     def generer_reponse(self, question: str) -> str:
#         """
#         Génère une réponse à la question de l'utilisateur.

#         Args:
#             question: La question de l'utilisateur

#         Returns:
#             str: La réponse générée
#         """
#         self.stats["questions_posees"] += 1
#         question_corrigee = preprocess_question(question)
#         if len(question_corrigee.split()) < 3:
#             question_corrigee = f"je veux savoir {question_corrigee}"

#         theme = self.detecter_intention(question_corrigee)
#         print(f"📂 Section détectée : {theme.replace('_', ' ').title()}")

#         reponses, scores = self.rechercher_reponse(question_corrigee, theme)
#         if not reponses:
#             self.stats["incompris"] += 1
#             self.logger.enregistrer_question(question, section_detectee=theme)
#             return "❌ Aucune réponse trouvée pour votre question."

#         best_score = float(scores[0])
#         best_reponse = reponses[0]

#         if best_score >= self.seuil_confiance_haute:
#             self.stats["reponses_directes"] += 1
#             return f"🎯 {best_reponse}"
#         elif best_score >= self.seuil_confiance_basse:
#             self.stats["suggestions"] += 1
#             suggestions = [r for r, s in zip(reponses, scores) if s >= self.seuil_confiance_basse][:2]
#             return "🤔 Voici quelques suggestions :\n" + "\n".join(f"  - {s}" for s in suggestions)
#         else:
#             self.stats["incompris"] += 1
#             self.logger.enregistrer_question(question, section_detectee=theme)
#             return "❌ Je n'ai pas compris votre question. Pouvez-vous la reformuler SVP ?"

#     def initialiser(self, rebuild_index: bool = False) -> None:
#         """
#         Initialise complètement le chatbot (modèles, données, index).

#         Args:
#             rebuild_index: Si True, force la reconstruction des index FAISS
#         """
#         print("🚀 Initialisation du Chatbot RH Nestlé...")
#         self.charger_modeles()
#         self.charger_donnees_faq()
#         self.creer_index_faiss(force=rebuild_index)
#         print("✅ Chatbot prêt à répondre !")

#     def afficher_statistiques(self) -> None:
#         """
#         Affiche les statistiques de la session en cours.
#         """
#         print("\n📊 STATISTIQUES DE SESSION :")
#         for key, value in self.stats.items():
#             print(f"  {key.replace('_', ' ').capitalize()} : {value}")
#         if self.stats["questions_posees"] > 0:
#             taux_comprehension = (
#                 (self.stats["reponses_directes"] + self.stats["suggestions"]) /
#                 self.stats["questions_posees"]
#             ) * 100
#             print(f"  Taux de compréhension : {taux_comprehension:.1f}%")

#     def executer(self) -> None:
#         """
#         Lance la boucle interactive du chatbot.
#         """
#         print("💬 Bienvenue sur le chatbot RH Nestlé")
#         print("   Tapez 'exit' ou 'quit' pour quitter\n")
#         while True:
#             try:
#                 question = input("Vous: ").strip()
#                 if question.lower() in ["exit", "quit"]:
#                     print("👋 Merci d'avoir utilisé le chatbot RH.")
#                     self.afficher_statistiques()
#                     break
#                 if not question:
#                     print("⚠️ Veuillez poser une question.")
#                     continue

#                 reponse = self.generer_reponse(question)
#                 print(f"Bot: {reponse}\n")
#             except KeyboardInterrupt:
#                 print("\n👋 Interruption manuelle détectée.")
#                 self.afficher_statistiques()
#                 break
#             except Exception as e:
#                 print(f"❌ Erreur inattendue : {e}")
#                 continue

# def main() -> None:
#     """
#     Point d'entrée principal du programme.
#     """
#     try:
#         chatbot = ChatbotRH()
#         chatbot.initialiser()
#         chatbot.executer()
#     except Exception as e:
#         print(f"❌ Erreur fatale : {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


"""
Chatbot RH Nestlé - Assistant conversationnel intelligent
Auteur: Système de chatbot RH
Description: Chatbot utilisant BERT et FAISS pour répondre aux questions RH avec correction orthographique.
"""

import json
import os
import sys
import joblib
import faiss
import numpy as np
from typing import Tuple, List, Dict, Any
from spellchecker import preprocess_question
from utils import charger_donnees_json
from logger import QuestionLogger
from sentence_transformers import SentenceTransformer


class ChatbotRH:
    """
    Chatbot RH utilisant BERT et FAISS pour répondre aux questions des employés.
    
    Attributes:
        model_path (str): Chemin vers le modèle BERT
        intent_model_path (str): Chemin vers le modèle de classification d'intentions
        data_path (str): Chemin vers les données FAQ
        log_path (str): Chemin vers le fichier de log
        seuil_confiance_haute (float): Seuil pour réponses directes
        seuil_confiance_basse (float): Seuil pour suggestions
    """
    
    def __init__(
        self,
        model_path: str = "model/sentence_bert_model.pkl",
        intent_model_path: str = "model/intent_classifier.pkl",
        data_path: str = "data/Nestle-HR-FAQ.json",
        log_path: str = "log/incompris.json",
        seuil_confiance_haute: float = 0.75,
        seuil_confiance_basse: float = 0.5,
    ):
        """
        Initialise le chatbot RH avec les paramètres spécifiés.
        
        Args:
            model_path: Chemin vers le modèle BERT
            intent_model_path: Chemin vers le modèle de classification d'intentions
            data_path: Chemin vers les données FAQ
            log_path: Chemin vers le fichier de log
            seuil_confiance_haute: Seuil pour réponses directes (défaut: 0.75)
            seuil_confiance_basse: Seuil pour suggestions (défaut: 0.5)
        """
        self.model_path = model_path
        self.intent_model_path = intent_model_path
        self.data_path = data_path
        self.log_path = log_path
        self.seuil_confiance_haute = seuil_confiance_haute
        self.seuil_confiance_basse = seuil_confiance_basse
        
        # Attributs initialisés après chargement
        self.model = None
        self.intent_model = None
        self.logger = None
        self.theme_indexes: Dict[str, Tuple[List[str], List[str], Any]] = {}
        self.full_data: Dict[str, List[Dict[str, str]]] = {}
        
        # Statistiques de session
        self.stats = {
            "questions_posees": 0,
            "reponses_directes": 0,
            "suggestions": 0,
            "incompris": 0,
        }

    def charger_modeles(self) -> None:
        """
        Charge les modèles BERT et de classification d'intentions.
        
        Raises:
            FileNotFoundError: Si les fichiers de modèles n'existent pas
            Exception: Si le chargement échoue
        """
        print("📦 Chargement des modèles...")
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle BERT non trouvé : {self.model_path}")
            
            if not os.path.exists(self.intent_model_path):
                raise FileNotFoundError(f"Modèle d'intention non trouvé : {self.intent_model_path}")
            
            self.model = joblib.load(self.model_path)
            self.intent_model = joblib.load(self.intent_model_path)
            self.logger = QuestionLogger(self.log_path)
            
            print("✅ Modèles chargés avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles : {e}")
            sys.exit(1)

    def charger_donnees_faq(self) -> None:
        """
        Charge les données FAQ depuis le fichier JSON.
        
        Raises:
            FileNotFoundError: Si le fichier FAQ n'existe pas
            json.JSONDecodeError: Si le fichier JSON est malformé
        """
        print("📂 Chargement des données FAQ...")
        
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Fichier FAQ non trouvé : {self.data_path}")
            
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.full_data = data["faq"]
            print(f"✅ {len(self.full_data)} thèmes RH chargés")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Erreur lors du chargement des données FAQ : {e}")
            sys.exit(1)

    def creer_index_faiss(self, force: bool = False) -> None:
        """
        Crée ou recharge les index FAISS pour chaque thème RH
        et les stocke dans le dossier model/indexes/
        
        Args:
            force: Si True, force la reconstruction des index (supprime le cache)
        
        Raises:
            Exception: Si la création/chargement des index échoue
        """
        dossier_index = "model/indexes"
        os.makedirs(dossier_index, exist_ok=True)
        
        # Mode REBUILD : suppression des anciens index
        if force:
            print("♻️ Mode REBUILD : suppression des anciens index...")
            for fname in os.listdir(dossier_index):
                os.remove(os.path.join(dossier_index, fname))
        
        print("📌 Chargement ou création des index FAISS par thème...")
        
        try:
            for theme, qa_list in self.full_data.items():
                questions = [item["question"] for item in qa_list]
                reponses = [item["response"] for item in qa_list]
                
                # Chemins des fichiers de cache
                index_path = os.path.join(dossier_index, f"{theme}.index")
                question_path = os.path.join(dossier_index, f"{theme}_questions.json")
                reponse_path = os.path.join(dossier_index, f"{theme}_reponses.json")
                
                # Tentative de chargement depuis le cache (sauf si force=True)
                if (not force and 
                    os.path.exists(index_path) and 
                    os.path.exists(question_path) and 
                    os.path.exists(reponse_path)):
                    
                    try:
                        # Charge l'index FAISS depuis le cache
                        index = faiss.read_index(index_path)
                        
                        # Charge les questions et réponses depuis le cache
                        with open(question_path, "r", encoding="utf-8") as f:
                            questions = json.load(f)
                        with open(reponse_path, "r", encoding="utf-8") as f:
                            reponses = json.load(f)
                        
                        print(f"🔁 Index FAISS chargé depuis cache pour : {theme}")
                        
                    except Exception as e:
                        print(f"⚠️ Erreur chargement index FAISS [{theme}], réencodage forcé... → {e}")
                        # Création forcée en cas d'erreur de chargement
                        self._creer_nouvel_index(theme, questions, reponses, index_path, question_path, reponse_path)
                        index = faiss.read_index(index_path)
                
                else:
                    # Création d'un nouvel index
                    print(f"➕ Création de l'index FAISS pour : {theme}")
                    self._creer_nouvel_index(theme, questions, reponses, index_path, question_path, reponse_path)
                    index = faiss.read_index(index_path)
                
                # Stockage dans le dictionnaire des index
                self.theme_indexes[theme] = (questions, reponses, index)
            
            print(f"✅ {len(self.theme_indexes)} index FAISS disponibles.")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création/chargement des index FAISS : {e}")
            sys.exit(1)
    
    def _creer_nouvel_index(self, theme: str, questions: List[str], reponses: List[str], 
                           index_path: str, question_path: str, reponse_path: str) -> None:
        """
        Crée un nouvel index FAISS et sauvegarde les fichiers de cache.
        
        Args:
            theme: Nom du thème
            questions: Liste des questions
            reponses: Liste des réponses
            index_path: Chemin du fichier d'index FAISS
            question_path: Chemin du fichier des questions
            reponse_path: Chemin du fichier des réponses
        """
        try:
            # Encode les questions en vecteurs
            vectors = self.model.encode(questions, convert_to_numpy=True).astype("float32")
            
            # Crée l'index FAISS
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            
            # Sauvegarde l'index FAISS
            faiss.write_index(index, index_path)
            
            # Sauvegarde les questions et réponses
            with open(question_path, "w", encoding="utf-8") as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)
            with open(reponse_path, "w", encoding="utf-8") as f:
                json.dump(reponses, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ Erreur lors de la création de l'index pour {theme} : {e}")
            raise

    def detecter_intention(self, question: str) -> str:
        """
        Détecte l'intention/thème de la question posée.
        
        Args:
            question: La question de l'utilisateur
            
        Returns:
            str: Le thème détecté
        """
        try:
            return self.intent_model.predict([question])[0]
        except Exception as e:
            print(f"⚠️ Erreur détection intention : {e}")
            # Retourne le premier thème disponible en cas d'erreur
            return list(self.full_data.keys())[0]

    def rechercher_reponse(self, question: str, theme: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """
        Recherche les meilleures réponses pour une question dans un thème donné.
        
        Args:
            question: La question à traiter
            theme: Le thème dans lequel rechercher
            k: Nombre de réponses à retourner
            
        Returns:
            Tuple[List[str], List[float]]: Les réponses et leurs scores de confiance
        """
        if theme not in self.theme_indexes:
            print(f"⚠️ Thème '{theme}' non trouvé")
            return [], []
        
        questions, reponses, index = self.theme_indexes[theme]
        
        # Encode la question en vecteur
        vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
        
        # Recherche les k plus proches voisins
        D, I = index.search(vec, k=min(k, len(questions)))
        
        # Convertit les distances L2 en scores de similarité [0, 1]
        scores = 1 - D[0] / 2
        
        return [reponses[i] for i in I[0]], scores.tolist()

    def generer_reponse(self, question: str) -> str:
        """
        Génère une réponse à la question de l'utilisateur.
        
        Args:
            question: La question de l'utilisateur
            
        Returns:
            str: La réponse générée
        """
        # Préprocessing de la question
        question_corrigee = preprocess_question(question)
        
        # Enrichit les questions trop courtes
        if len(question_corrigee.split()) < 3:
            question_corrigee = f"je veux savoir {question_corrigee}"

        # Détecte l'intention/thème
        theme = self.detecter_intention(question_corrigee)
        print(f"📂 Section détectée : {theme.replace('_', ' ').title()}")
        
        # Recherche les réponses
        reponses, scores = self.rechercher_reponse(question_corrigee, theme, k=3)
        
        if not reponses:
            self.stats["incompris"] += 1
            self.logger.enregistrer_question(question, section_detectee=theme)
            return "❌ Aucune réponse trouvée pour votre question."
        
        best_score = float(scores[0])
        best_reponse = reponses[0]

        # Réponse directe avec haute confiance
        if best_score >= self.seuil_confiance_haute:
            self.stats["reponses_directes"] += 1
            return f"🎯 {best_reponse}"

        # Suggestions avec confiance moyenne
        elif best_score >= self.seuil_confiance_basse:
            self.stats["suggestions"] += 1
            suggestions = [
                r for r, s in zip(reponses, scores) 
                if s >= self.seuil_confiance_basse
            ]
            suggestions = suggestions[:2] if len(suggestions) > 1 else [best_reponse]
            
            return "🤔 Voici quelques suggestions :\n" + "\n".join(f"  - {s}" for s in suggestions)

        # Question non comprise
        else:
            self.stats["incompris"] += 1
            self.logger.enregistrer_question(question, section_detectee=theme)
            return "❌ Je n'ai pas compris votre question. Pouvez-vous la reformuler SVP ?"

    def initialiser(self, rebuild_index: bool = False) -> None:
        """
        Initialise complètement le chatbot (modèles, données, index).
        
        Args:
            rebuild_index: Si True, force la reconstruction des index FAISS
        """
        print("🚀 Initialisation du Chatbot RH Nestlé...")
        
        self.charger_modeles()
        self.charger_donnees_faq()
        self.creer_index_faiss(force=rebuild_index)
        
        print("✅ Chatbot prêt à répondre !")

    def afficher_statistiques(self) -> None:
        """
        Affiche les statistiques de la session en cours.
        """
        print("\n📊 STATISTIQUES DE SESSION :")
        for key, value in self.stats.items():
            print(f"  {key.replace('_', ' ').capitalize()} : {value}")
        
        if self.stats["questions_posees"] > 0:
            taux_comprehension = (
                (self.stats["reponses_directes"] + self.stats["suggestions"]) /
                self.stats["questions_posees"]
            ) * 100
            print(f"  Taux de compréhension : {taux_comprehension:.1f}%")

    def executer(self) -> None:
        """
        Lance la boucle interactive du chatbot.
        """
        print("💬 Bienvenue sur le chatbot RH Nestlé")
        print("   Tapez 'exit' ou 'quit' pour quitter\n")
        
        while True:
            try:
                question = input("Vous: ").strip()
                
                # Commandes de sortie
                if question.lower() in ["exit", "quit", "sortir", "quitter"]:
                    print("👋 Merci d'avoir utilisé le chatbot RH.")
                    self.afficher_statistiques()
                    break
                
                # Question vide
                if not question:
                    print("⚠️ Veuillez poser une question.")
                    continue

                # Traitement de la question
                self.stats["questions_posees"] += 1
                reponse = self.generer_reponse(question)
                print(f"Bot: {reponse}\n")

            except KeyboardInterrupt:
                print("\n👋 Interruption manuelle détectée.")
                self.afficher_statistiques()
                break
            except Exception as e:
                print(f"❌ Erreur inattendue : {e}")
                continue


def main():
    """
    Point d'entrée principal du programme.
    """
    try:
        chatbot = ChatbotRH()
        chatbot.initialiser()
        chatbot.executer()
    except Exception as e:
        print(f"❌ Erreur fatale : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()