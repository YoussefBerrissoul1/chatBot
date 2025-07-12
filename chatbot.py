"""
Chatbot RH Nestlé - Assistant conversationnel intelligent
Auteur: Système de chatbot RH
Description: Chatbot utilisant BERT et FAISS pour répondre aux questions RH
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# Imports locaux
from utils import charger_donnees_json
from logger import QuestionLogger


class ChatbotRH:
    """
    Chatbot RH intelligent utilisant BERT et FAISS pour la recherche sémantique
    
    Attributes:
        model: Modèle BERT pour l'encodage des phrases
        intent_model: Classificateur d'intentions pour détecter les thèmes
        logger: Logger pour enregistrer les questions non comprises
        theme_indexes: Index FAISS pour chaque thème RH
        full_data: Données complètes des FAQ
        seuil_confiance_haute: Seuil pour réponses directes
        seuil_confiance_basse: Seuil pour suggestions
    """
    
    def __init__(self, 
                model_path: str = "model/sentence_bert_model.pkl",
                intent_model_path: str = "model/intent_classifier.pkl",
                data_path: str = "data/Nestle-HR-FAQ.json",
                log_path: str = "log/incompris.json",
                seuil_confiance_haute: float = 0.75,
                seuil_confiance_basse: float = 0.5):
        """
        Initialise le chatbot RH
        
        Args:
            model_path (str): Chemin vers le modèle BERT
            intent_model_path (str): Chemin vers le classificateur d'intentions
            data_path (str): Chemin vers les données FAQ
            log_path (str): Chemin vers le fichier de log
            seuil_confiance_haute (float): Seuil pour réponses directes
            seuil_confiance_basse (float): Seuil pour suggestions
        """
        self.model_path = model_path
        self.intent_model_path = intent_model_path
        self.data_path = data_path
        self.log_path = log_path
        self.seuil_confiance_haute = seuil_confiance_haute
        self.seuil_confiance_basse = seuil_confiance_basse
        
        # Initialisation des composants
        self.model = None
        self.intent_model = None
        self.logger = None
        self.theme_indexes = {}
        self.full_data = {}
        
        # Statistiques de session
        self.stats = {
            "questions_posees": 0,
            "reponses_directes": 0,
            "suggestions": 0,
            "incompris": 0
        }
    
    def charger_modeles(self):
        """
        Charge les modèles de machine learning
        """
        print("📦 Chargement des modèles...")
        
        try:
            # Chargement du modèle BERT
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("✅ Modèle BERT chargé avec succès")
            else:
                print(f"❌ Modèle BERT introuvable : {self.model_path}")
                raise FileNotFoundError(f"Modèle BERT non trouvé : {self.model_path}")
            
            # Chargement du classificateur d'intentions
            if os.path.exists(self.intent_model_path):
                self.intent_model = joblib.load(self.intent_model_path)
                print("✅ Classificateur d'intentions chargé avec succès")
            else:
                print(f"❌ Classificateur introuvable : {self.intent_model_path}")
                raise FileNotFoundError(f"Classificateur non trouvé : {self.intent_model_path}")
            
            # Initialisation du logger
            self.logger = QuestionLogger(self.log_path)
            print("✅ Logger initialisé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles : {e}")
            raise
    
    def charger_donnees_faq(self):
        """
        Charge les données FAQ depuis le fichier JSON
        """
        print("📂 Chargement des données FAQ...")
        
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "faq" not in data:
                raise ValueError("Structure JSON invalide : clé 'faq' manquante")
            
            self.full_data = data["faq"]
            print(f"✅ {len(self.full_data)} thèmes RH chargés")
            
            # Affichage des thèmes disponibles
            print("📋 Thèmes disponibles :")
            for theme in self.full_data.keys():
                nb_questions = len(self.full_data[theme])
                theme_formate = theme.replace('_', ' ').title()
                print(f"   🔹 {theme_formate} ({nb_questions} questions)")
            
        except FileNotFoundError:
            print(f"❌ Fichier de données introuvable : {self.data_path}")
            raise
        except json.JSONDecodeError:
            print(f"❌ Format JSON invalide : {self.data_path}")
            raise
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            raise
    
    def creer_index_faiss(self):
        """
        Crée les index FAISS pour chaque thème RH
        """
        print("📌 Création des index FAISS pour chaque thème RH...")
        
        try:
            for theme, qa_list in self.full_data.items():
                print(f"🔧 Indexation du thème : {theme}")
                
                # Extraction des questions et réponses
                questions = [item["question"] for item in qa_list]
                reponses = [item["response"] for item in qa_list]
                
                # Encodage des questions avec BERT
                vectors = self.model.encode(questions, convert_to_numpy=True).astype("float32")
                
                # Création de l'index FAISS
                dimension = vectors.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(vectors)
                
                # Stockage de l'index et des données associées
                self.theme_indexes[theme] = (questions, reponses, index)
                
                print(f"   ✅ {len(questions)} questions indexées")
            
            print(f"🎯 {len(self.theme_indexes)} thèmes indexés avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de la création des index : {e}")
            raise
    
    def detecter_intention(self, question: str) -> str:
        """
        Détecte l'intention (thème) d'une question
        
        Args:
            question (str): Question de l'utilisateur
            
        Returns:
            str: Thème détecté
        """
        try:
            theme = self.intent_model.predict([question])[0]
            return theme
        except Exception as e:
            print(f"⚠️ Erreur de détection d'intention : {e}")
            # Retourne le premier thème par défaut
            return list(self.full_data.keys())[0]
    
    def rechercher_reponse(self, question: str, theme: str, k: int = 3) -> Tuple[str, float]:
        """
        Recherche la meilleure réponse pour une question dans un thème donné
        
        Args:
            question (str): Question de l'utilisateur
            theme (str): Thème détecté
            k (int): Nombre de réponses candidates à considérer
            
        Returns:
            Tuple[str, float]: Meilleure réponse et score de confiance
        """
        try:
            # Récupération des données du thème
            questions, reponses, index = self.theme_indexes[theme]
            
            # Encodage de la question
            vec = self.model.encode([question], convert_to_numpy=True).astype("float32")
            
            # Recherche des k plus proches voisins
            distances, indices = index.search(vec, k=min(k, len(questions)))
            
            # Conversion des distances L2 en scores de similarité [0, 1]
            similarities = 1 - distances[0] / 2
            
            # Sélection de la meilleure réponse
            best_idx_among_k = np.argmax(similarities)
            best_global_idx = int(indices[0][best_idx_among_k])
            best_score = float(similarities[best_idx_among_k])
            
            return reponses[best_global_idx], best_score
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la recherche : {e}")
            return "Erreur lors de la recherche de réponse.", 0.0
    
    def generer_reponse(self, question: str) -> str:
        """
        Génère une réponse appropriée selon le score de confiance
        
        Args:
            question (str): Question de l'utilisateur
            
        Returns:
            str: Réponse formatée
        """
        # Détection de l'intention
        theme = self.detecter_intention(question)
        theme_formate = theme.replace('_', ' ').title()
        
        print(f"📂 Section détectée : {theme_formate}")
        
        # Recherche de la meilleure réponse
        reponse, score = self.rechercher_reponse(question, theme)
        
        # Génération de la réponse selon le score de confiance
        if score >= self.seuil_confiance_haute:
            # Réponse directe avec haute confiance
            self.stats["reponses_directes"] += 1
            return f"🎯 {reponse}"
            
        elif score >= self.seuil_confiance_basse:
            # Suggestion avec confiance modérée
            self.stats["suggestions"] += 1
            return f"🤔 Peut-être vouliez-vous dire :\n   {reponse}"
            
        else:
            # Question non comprise
            self.stats["incompris"] += 1
            self.logger.enregistrer_question(question, section_detectee=theme)
            return "❌ Je n'ai pas compris votre question. Pourriez-vous la reformuler ?\n" \
                   "💡 Astuce : Essayez d'être plus spécifique ou utilisez des mots-clés liés aux RH."
    
    def afficher_aide(self):
        """
        Affiche l'aide et les commandes disponibles
        """
        print("\n" + "="*60)
        print("🆘 AIDE - CHATBOT RH NESTLÉ")
        print("="*60)
        print("📝 Commandes disponibles :")
        print("   • 'exit' ou 'quit' : Quitter le chatbot")
        print("   • 'help' ou 'aide' : Afficher cette aide")
        print("   • 'stats' : Afficher les statistiques de session")
        print("   • 'themes' : Lister les thèmes disponibles")
        print("\n💡 Conseils pour de meilleures réponses :")
        print("   • Soyez spécifique dans vos questions")
        print("   • Utilisez des mots-clés liés aux RH")
        print("   • Posez une question à la fois")
        print("="*60 + "\n")
    
    def afficher_themes(self):
        """
        Affiche la liste des thèmes disponibles
        """
        print("\n📋 THÈMES RH DISPONIBLES :")
        print("-" * 40)
        for theme in self.full_data.keys():
            nb_questions = len(self.full_data[theme])
            theme_formate = theme.replace('_', ' ').title()
            print(f"🔹 {theme_formate} ({nb_questions} questions)")
        print("-" * 40 + "\n")
    
    def afficher_statistiques(self):
        """
        Affiche les statistiques de la session
        """
        print("\n📊 STATISTIQUES DE SESSION :")
        print("-" * 40)
        print(f"❓ Questions posées      : {self.stats['questions_posees']}")
        print(f"✅ Réponses directes     : {self.stats['reponses_directes']}")
        print(f"🤔 Suggestions données   : {self.stats['suggestions']}")
        print(f"❌ Questions non comprises : {self.stats['incompris']}")
        
        if self.stats['questions_posees'] > 0:
            taux_comprehension = ((self.stats['reponses_directes'] + self.stats['suggestions']) / 
                                 self.stats['questions_posees']) * 100
            print(f"📈 Taux de compréhension : {taux_comprehension:.1f}%")
        
        print("-" * 40 + "\n")
    
    def initialiser(self):
        """
        Initialise tous les composants du chatbot
        """
        print("🚀 Initialisation du Chatbot RH Nestlé...")
        print("="*60)
        
        try:
            # Chargement des modèles
            self.charger_modeles()
            
            # Chargement des données FAQ
            self.charger_donnees_faq()
            
            # Création des index FAISS
            self.creer_index_faiss()
            
            print("="*60)
            print("✅ Chatbot RH Nestlé prêt à l'emploi !")
            print("💡 Tapez 'help' pour voir les commandes disponibles")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation : {e}")
            raise
    
    def executer(self):
        """
        Lance la boucle conversationnelle du chatbot
        """
        print("💬 Chatbot RH Nestlé")
        print("💡 Tapez 'help' pour l'aide, 'exit' pour quitter\n")
        
        while True:
            try:
                # Saisie de la question
                question = input("Vous: ").strip()
                
                # Vérification des commandes système
                if question.lower() in ["exit", "quit"]:
                    print("👋 Merci d'avoir utilisé le chatbot RH Nestlé !")
                    self.afficher_statistiques()
                    break
                
                elif question.lower() in ["help", "aide"]:
                    self.afficher_aide()
                    continue
                
                elif question.lower() == "stats":
                    self.afficher_statistiques()
                    continue
                
                elif question.lower() == "themes":
                    self.afficher_themes()
                    continue
                
                elif not question:
                    print("⚠️ Veuillez poser une question.")
                    continue
                
                # Traitement de la question
                self.stats["questions_posees"] += 1
                reponse = self.generer_reponse(question)
                print(f"Bot: {reponse}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"⚠️ Erreur inattendue : {e}")
                continue


def main():
    """
    Fonction principale pour lancer le chatbot
    """
    try:
        # Création et initialisation du chatbot
        chatbot = ChatbotRH()
        chatbot.initialiser()
        
        # Lancement de la conversation
        chatbot.executer()
        
    except KeyboardInterrupt:
        print("\n👋 Programme interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        print("🔧 Vérifiez votre configuration et réessayez")


if __name__ == "__main__":
    main()