"""
Script d'évaluation et d'optimisation du chatbot RH
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from chatbot_optimized import ChatbotRHOptimise


class EvaluateurChatbot:
    """
    Classe pour évaluer et optimiser les performances du chatbot
    """
    
    def __init__(self, chatbot: ChatbotRHOptimise):
        self.chatbot = chatbot
        self.questions_test = []
        self.themes_attendus = []
        self.resultats = {}
    
    def creer_dataset_test(self) -> None:
        """Crée un dataset de test à partir des données existantes"""
        print("🔄 Création du dataset de test...")
        
        # Prendre 20% des données pour le test
        indices = np.random.choice(
            len(self.chatbot.questions), 
            size=max(1, len(self.chatbot.questions) // 5), 
            replace=False
        )
        
        for i in indices:
            self.questions_test.append(self.chatbot.questions[i])
            self.themes_attendus.append(self.chatbot.themes[i])
        
        print(f"✅ {len(self.questions_test)} questions de test créées")
    
    def evaluer_classification_themes(self) -> dict:
        """Évalue la performance de classification des thèmes"""
        print("🎯 Évaluation de la classification des thèmes...")
        
        themes_predits = []
        for question in self.questions_test:
            theme_predit = self.chatbot.detecter_theme(question)
            themes_predits.append(theme_predit)
        
        # Calculer les métriques
        accuracy = accuracy_score(self.themes_attendus, themes_predits)
        rapport = classification_report(
            self.themes_attendus, 
            themes_predits, 
            output_dict=True
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.chatbot.classifier, 
            self.chatbot.questions, 
            self.chatbot.themes, 
            cv=5
        )
        
        resultats = {
            'accuracy': accuracy,
            'rapport': rapport,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"✅ CV Score moyen: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return resultats
    
    def evaluer_qualite_reponses(self) -> dict:
        """Évalue la qualité des réponses générées"""
        print("🔍 Évaluation de la qualité des réponses...")
        
        scores_confiance = []
        reponses_correctes = 0
        
        for question in self.questions_test:
            reponse, score = self.chatbot.trouver_meilleure_reponse(question)
            scores_confiance.append(score)
            
            if score >= self.chatbot.seuil_confiance:
                reponses_correctes += 1
        
        resultats = {
            'score_confiance_moyen': np.mean(scores_confiance),
            'score_confiance_std': np.std(scores_confiance),
            'taux_reponses_correctes': reponses_correctes / len(self.questions_test),
            'scores_confiance': scores_confiance
        }
        
        print(f"✅ Score de confiance moyen: {resultats['score_confiance_moyen']:.3f}")
        print(f"✅ Taux de réponses correctes: {resultats['taux_reponses_correctes']:.3f}")
        
        return resultats
    
    def optimiser_seuil_confiance(self) -> float:
        """Trouve le seuil de confiance optimal"""
        print("⚙️ Optimisation du seuil de confiance...")
        
        seuils = np.arange(0.1, 0.9, 0.05)
        meilleurs_resultats = []
        
        for seuil in seuils:
            self.chatbot.seuil_confiance = seuil
            reponses_correctes = 0
            
            for question in self.questions_test:
                reponse, score = self.chatbot.trouver_meilleure_reponse(question)
                if score >= seuil:
                    reponses_correctes += 1
            
            taux_reussite = reponses_correctes / len(self.questions_test)
            meilleurs_resultats.append(taux_reussite)
        
        # Trouver le seuil optimal
        meilleur_index = np.argmax(meilleurs_resultats)
        seuil_optimal = seuils[meilleur_index]
        
        print(f"✅ Seuil optimal trouvé: {seuil_optimal:.3f}")
        print(f"✅ Taux de réussite avec ce seuil: {meilleurs_resultats[meilleur_index]:.3f}")
        
        return seuil_optimal
    
    def generer_rapport_complet(self) -> None:
        """Génère un rapport complet d'évaluation"""
        print("📊 Génération du rapport complet...")
        
        # Évaluations
        resultats_classification = self.evaluer_classification_themes()
        resultats_reponses = self.evaluer_qualite_reponses()
        seuil_optimal = self.optimiser_seuil_confiance()
        
        # Préparer le rapport
        rapport = {
            "evaluation_date": "2024-01-15",
            "dataset_test": {
                "nombre_questions": len(self.questions_test),
                "nombre_themes": len(set(self.themes_attendus))
            },
            "classification_themes": {
                "accuracy": resultats_classification['accuracy'],
                "cv_score_moyen": resultats_classification['cv_mean'],
                "cv_score_std": resultats_classification['cv_std']
            },
            "qualite_reponses": {
                "score_confiance_moyen": resultats_reponses['score_confiance_moyen'],
                "taux_reponses_correctes": resultats_reponses['taux_reponses_correctes']
            },
            "optimisation": {
                "seuil_optimal": seuil_optimal,
                "seuil_actuel": self.chatbot.seuil_confiance
            },
            "recommandations": self.generer_recommandations(resultats_classification, resultats_reponses)
        }
        
        # Sauvegarder le rapport
        with open("evaluation_rapport.json", 'w', encoding='utf-8') as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)
        
        print("✅ Rapport sauvegardé dans 'evaluation_rapport.json'")
        self.afficher_resume_rapport(rapport)
    
    def generer_recommandations(self, resultats_classification: dict, resultats_reponses: dict) -> list:
        """Génère des recommandations d'amélioration"""
        recommandations = []
        
        if resultats_classification['accuracy'] < 0.8:
            recommandations.append("Améliorer la classification des thèmes avec plus de données d'entraînement")
        
        if resultats_reponses['score_confiance_moyen'] < 0.5:
            recommandations.append("Enrichir les données FAQ avec plus de variantes de questions")
        
        if resultats_reponses['taux_reponses_correctes'] < 0.7:
            recommandations.append("Ajuster le seuil de confiance ou améliorer le préprocessing")
        
        if len(recommandations) == 0:
            recommandations.append("Performance satisfaisante - continuer le monitoring")
        
        return recommandations
    
    def afficher_resume_rapport(self, rapport: dict) -> None:
        """Affiche un résumé du rapport d'évaluation"""
        print("\n" + "="*60)
        print("📋 RÉSUMÉ DU RAPPORT D'ÉVALUATION")
        print("="*60)
        
        print(f"📊 Dataset de test: {rapport['dataset_test']['nombre_questions']} questions")
        print(f"🎯 Accuracy classification: {rapport['classification_themes']['accuracy']:.3f}")
        print(f"📈 Score CV moyen: {rapport['classification_themes']['cv_score_moyen']:.3f}")
        print(f"🔍 Score confiance moyen: {rapport['qualite_reponses']['score_confiance_moyen']:.3f}")
        print(f"✅ Taux réponses correctes: {rapport['qualite_reponses']['taux_reponses_correctes']:.3f}")
        print(f"⚙️ Seuil optimal: {rapport['optimisation']['seuil_optimal']:.3f}")
        
        print("\n🎯 RECOMMANDATIONS:")
        for i, rec in enumerate(rapport['recommandations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*60)
    
    def executer_evaluation_complete(self) -> None:
        """Exécute une évaluation complète du chatbot"""
        print("🚀 Démarrage de l'évaluation complète...")
        
        self.creer_dataset_test()
        self.generer_rapport_complet()
        
        print("✅ Évaluation terminée avec succès!")


def main():
    """Point d'entrée principal pour l'évaluation"""
    print("🔬 Évaluateur de Performance - Chatbot RH")
    print("="*50)
    
    # Initialiser le chatbot
    chatbot = ChatbotRHOptimise()
    chatbot.initialiser()
    
    # Créer l'évaluateur
    evaluateur = EvaluateurChatbot(chatbot)
    
    # Exécuter l'évaluation
    evaluateur.executer_evaluation_complete()


if __name__ == "__main__":
    main()