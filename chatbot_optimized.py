"""
Chatbot RH Nestlé Optimisé - Version simplifiée et performante
Description: Chatbot utilisant TF-IDF et similarité cosinus pour des réponses précises
"""

import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ChatbotRHOptimise:
    """
    Chatbot RH optimisé avec TF-IDF et classification Naive Bayes
    """
    
    def __init__(self, data_path: str = "data/Nestle-HR-FAQ.json"):
        self.data_path = data_path
        self.seuil_confiance = 0.3  # Seuil plus bas pour plus de flexibilité
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # Garder tous les mots pour le français
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('nb', MultinomialNB(alpha=0.1))
        ])
        
        # Données
        self.questions = []
        self.reponses = []
        self.themes = []
        self.tfidf_matrix = None
        self.theme_data = {}
        
        # Statistiques
        self.stats = {
            "questions_posees": 0,
            "reponses_trouvees": 0,
            "questions_incomprises": 0
        }
    
    def nettoyer_texte(self, texte: str) -> str:
        """Nettoie et normalise le texte"""
        if not texte:
            return ""
        
        texte = texte.lower()
        texte = re.sub(r'[^\w\s]', ' ', texte)  # Supprime la ponctuation
        texte = re.sub(r'\s+', ' ', texte)      # Normalise les espaces
        return texte.strip()
    
    def charger_donnees(self) -> None:
        """Charge les données FAQ depuis le fichier JSON"""
        print("📂 Chargement des données FAQ...")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraire les questions, réponses et thèmes
            for theme, items in data["faq"].items():
                self.theme_data[theme] = []
                for item in items:
                    question = self.nettoyer_texte(item["question"])
                    reponse = item["response"]
                    
                    self.questions.append(question)
                    self.reponses.append(reponse)
                    self.themes.append(theme)
                    self.theme_data[theme].append({
                        "question": question,
                        "response": reponse
                    })
            
            print(f"✅ {len(self.questions)} questions chargées dans {len(self.theme_data)} thèmes")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            raise
    
    def entrainer_modeles(self) -> None:
        """Entraîne les modèles TF-IDF et de classification"""
        print("🎯 Entraînement des modèles...")
        
        # Entraîner le vectorizer TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        
        # Entraîner le classificateur de thèmes
        self.classifier.fit(self.questions, self.themes)
        
        print("✅ Modèles entraînés avec succès")
    
    def detecter_theme(self, question: str) -> str:
        """Détecte le thème de la question"""
        try:
            question_nettoyee = self.nettoyer_texte(question)
            theme_predit = self.classifier.predict([question_nettoyee])[0]
            return theme_predit
        except Exception:
            # Retour vers le premier thème en cas d'erreur
            return list(self.theme_data.keys())[0]
    
    def trouver_meilleure_reponse(self, question: str) -> Tuple[str, float]:
        """
        Trouve la meilleure réponse à une question donnée
        
        Returns:
            Tuple[str, float]: (réponse, score de confiance)
        """
        question_nettoyee = self.nettoyer_texte(question)
        
        # Vectoriser la question
        question_vector = self.vectorizer.transform([question_nettoyee])
        
        # Calculer la similarité avec toutes les questions
        similarites = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Trouver l'index de la meilleure similarité
        meilleur_index = np.argmax(similarites)
        meilleur_score = similarites[meilleur_index]
        
        return self.reponses[meilleur_index], meilleur_score
    
    def generer_reponse(self, question: str) -> str:
        """Génère une réponse à la question de l'utilisateur"""
        self.stats["questions_posees"] += 1
        
        if not question.strip():
            return "⚠️ Veuillez poser une question."
        
        # Détecter le thème
        theme = self.detecter_theme(question)
        
        # Trouver la meilleure réponse
        reponse, score = self.trouver_meilleure_reponse(question)
        
        # Afficher le thème détecté
        print(f"📁 Thème détecté: {theme.replace('_', ' ').title()}")
        print(f"🎯 Score de confiance: {score:.2f}")
        
        if score >= self.seuil_confiance:
            self.stats["reponses_trouvees"] += 1
            return f"✅ {reponse}"
        else:
            self.stats["questions_incomprises"] += 1
            self.enregistrer_question_incomprise(question, theme)
            return (f"❌ Je n'ai pas bien compris votre question (score: {score:.2f}).\n"
                   f"Voici quand même ma meilleure suggestion :\n{reponse}\n\n"
                   f"💡 Essayez de reformuler votre question ou d'être plus spécifique.")
    
    def enregistrer_question_incomprise(self, question: str, theme: str) -> None:
        """Enregistre les questions mal comprises"""
        log_path = "log/questions_incomprises.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Charger le fichier existant ou créer une nouvelle structure
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"questions": []}
        
        # Ajouter la nouvelle entrée
        data["questions"].append({
            "question": question,
            "theme_detecte": theme,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Sauvegarder
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def sauvegarder_modeles(self) -> None:
        """Sauvegarde les modèles entraînés"""
        os.makedirs("model", exist_ok=True)
        
        # Sauvegarder le vectorizer
        with open("model/vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Sauvegarder le classificateur
        with open("model/classifier.pkl", 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Sauvegarder les données
        with open("model/chatbot_data.pkl", 'wb') as f:
            pickle.dump({
                'questions': self.questions,
                'reponses': self.reponses,
                'themes': self.themes,
                'tfidf_matrix': self.tfidf_matrix
            }, f)
        
        print("💾 Modèles sauvegardés avec succès")
    
    def charger_modeles(self) -> bool:
        """Charge les modèles sauvegardés"""
        try:
            with open("model/vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open("model/classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            
            with open("model/chatbot_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.questions = data['questions']
                self.reponses = data['reponses']
                self.themes = data['themes']
                self.tfidf_matrix = data['tfidf_matrix']
            
            print("✅ Modèles chargés depuis le cache")
            return True
        except Exception as e:
            print(f"⚠️ Impossible de charger les modèles depuis le cache : {e}")
            return False
    
    def initialiser(self, force_retrain: bool = False) -> None:
        """Initialise le chatbot"""
        print("🚀 Initialisation du Chatbot RH Nestlé...")
        
        # Essayer de charger les modèles existants
        if not force_retrain and self.charger_modeles():
            print("✅ Chatbot initialisé avec les modèles sauvegardés")
            return
        
        # Sinon, charger les données et entraîner
        self.charger_donnees()
        self.entrainer_modeles()
        self.sauvegarder_modeles()
        
        print("✅ Chatbot initialisé et prêt à répondre!")
    
    def afficher_statistiques(self) -> None:
        """Affiche les statistiques de la session"""
        print("\n📊 STATISTIQUES DE SESSION:")
        print(f"  Questions posées: {self.stats['questions_posees']}")
        print(f"  Réponses trouvées: {self.stats['reponses_trouvees']}")
        print(f"  Questions incomprises: {self.stats['questions_incomprises']}")
        
        if self.stats['questions_posees'] > 0:
            taux_reussite = (self.stats['reponses_trouvees'] / self.stats['questions_posees']) * 100
            print(f"  Taux de réussite: {taux_reussite:.1f}%")
    
    def tester_performance(self) -> None:
        """Teste la performance du chatbot avec quelques questions"""
        questions_test = [
            "Comment demander des congés?",
            "Quelle est la politique de télétravail?",
            "Comment signaler un problème?",
            "Où trouver mes fiches de paie?",
            "Qui contacter pour la formation?"
        ]
        
        print("\n🧪 TEST DE PERFORMANCE:")
        print("-" * 50)
        
        for question in questions_test:
            reponse = self.generer_reponse(question)
            print(f"Q: {question}")
            print(f"R: {reponse[:100]}...")
            print("-" * 50)
    
    def executer(self) -> None:
        """Lance la boucle interactive du chatbot"""
        print("\n💬 Chatbot RH Nestlé - Version Optimisée")
        print("   Tapez 'exit', 'quit' ou 'test' pour les options spéciales\n")
        
        while True:
            try:
                question = input("Vous: ").strip()
                
                if question.lower() in ['exit', 'quit', 'sortir']:
                    print("👋 Merci d'avoir utilisé le chatbot RH!")
                    self.afficher_statistiques()
                    break
                
                if question.lower() == 'test':
                    self.tester_performance()
                    continue
                
                if question.lower() == 'stats':
                    self.afficher_statistiques()
                    continue
                
                if not question:
                    print("⚠️ Veuillez poser une question.")
                    continue
                
                reponse = self.generer_reponse(question)
                print(f"Bot: {reponse}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Arrêt du chatbot.")
                self.afficher_statistiques()
                break
            except Exception as e:
                print(f"❌ Erreur: {e}")
                continue


def main():
    """Point d'entrée principal"""
    try:
        chatbot = ChatbotRHOptimise()
        chatbot.initialiser()
        chatbot.executer()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")


if __name__ == "__main__":
    main()