"""
test_chatbot.py
---------------
Script de test automatique du chatbot RH Nestlé avec timer et rebuild
"""

import time
import argparse
from chatbot import ChatbotRH


def test_questions(chatbot: ChatbotRH, questions: list) -> dict:
    """
    Teste une liste de questions et retourne les statistiques.
    
    Args:
        chatbot: Instance du chatbot RH
        questions: Liste des questions à tester
        
    Returns:
        dict: Statistiques des tests
    """
    total = len(questions)
    bien_comprises = 0
    reponses_directes = 0
    suggestions = 0
    incomprises = 0
    
    print("\n📊 Début des tests...\n")
    
    for i, q in enumerate(questions, 1):
        print(f"🔹 Q{i}: {q}")
        reponse = chatbot.generer_reponse(q)
        print(f"🔸 Réponse: {reponse}\n")
        
        # Classification des réponses
        if "🎯" in reponse:
            bien_comprises += 1
            reponses_directes += 1
        elif "🤔" in reponse:
            bien_comprises += 1
            suggestions += 1
        elif "❌" in reponse:
            incomprises += 1
    
    # Statistiques
    stats = {
        "total": total,
        "bien_comprises": bien_comprises,
        "reponses_directes": reponses_directes,
        "suggestions": suggestions,
        "incomprises": incomprises,
        "taux_comprehension": (bien_comprises / total) * 100 if total > 0 else 0
    }
    
    return stats


def afficher_resultats(stats: dict, temps_demarrage: float) -> None:
    """
    Affiche les résultats détaillés des tests.
    
    Args:
        stats: Statistiques des tests
        temps_demarrage: Temps de démarrage du chatbot
    """
    print("=" * 50)
    print("📋 RÉSULTATS DES TESTS")
    print("=" * 50)
    print(f"⏱️  Temps de démarrage     : {temps_demarrage:.2f} secondes")
    print(f"📊 Questions testées      : {stats['total']}")
    print(f"🎯 Réponses directes      : {stats['reponses_directes']}")
    print(f"🤔 Suggestions proposées  : {stats['suggestions']}")
    print(f"❌ Questions incomprises  : {stats['incomprises']}")
    print(f"✅ Questions comprises    : {stats['bien_comprises']}/{stats['total']}")
    print(f"📈 Taux de compréhension  : {stats['taux_comprehension']:.1f}%")
    
    # Évaluation de la performance
    if stats['taux_comprehension'] >= 80:
        print("🏆 Performance : EXCELLENTE")
    elif stats['taux_comprehension'] >= 60:
        print("👍 Performance : BONNE")
    elif stats['taux_comprehension'] >= 40:
        print("⚠️  Performance : MOYENNE")
    else:
        print("⚠️  Performance : À AMÉLIORER")


def main():
    """
    Fonction principale du script de test.
    """
    parser = argparse.ArgumentParser(
        description="Test automatique du chatbot RH Nestlé"
    )
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="Reconstruire les index FAISS"
    )
    parser.add_argument(
        "--questions-file", 
        type=str, 
        help="Fichier JSON contenant les questions de test"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Affichage détaillé des réponses"
    )
    
    args = parser.parse_args()
    
    # Questions de test par défaut
    questions_defaut = [
        "Comment postuler chez Nestlé ?",
        "Quels sont les critères pour une alternance ?",
        "Quels sont les délais de réponse après une candidature ?",
        "Comment modifier ma candidature ?",
        "Est-ce que je peux passer plusieurs entretiens ?",
        "Comment puis-je poser mes congés ?",
        "Quels sont les avantages d'un contrat CDI ?",
        "Est-ce que je peux avoir un bulletin de paie imprimé ?",
        "Que faire en cas de retard de paie ?",
        "Combien de jours de télétravail sont autorisés ?",
        "Nestlé propose-t-il une mutuelle santé ?",
        "Mon salaire est-il versé à la fin du mois ?",
        "Puis-je faire une demande d'avancement ?",
        "Y a-t-il une politique de travail hybride ?",
        "Comment faire si je suis victime de harcèlement ?"
    ]
    
    # Chargement des questions depuis un fichier si spécifié
    if args.questions_file:
        try:
            import json
            with open(args.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions = data.get('questions', questions_defaut)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du fichier : {e}")
            print("📋 Utilisation des questions par défaut...")
            questions = questions_defaut
    else:
        questions = questions_defaut
    
    # Initialisation du chatbot avec timer
    print("🚀 Lancement du chatbot RH avec timer...")
    if args.rebuild:
        print("♻️ Mode REBUILD activé - Reconstruction des index FAISS")
    
    t0 = time.time()
    
    try:
        chatbot = ChatbotRH()
        chatbot.initialiser(rebuild_index=args.rebuild)
        
        t1 = time.time()
        temps_demarrage = t1 - t0
        
        print(f"\n✅ Démarrage terminé en {temps_demarrage:.2f} secondes")
        
        # Exécution des tests
        stats = test_questions(chatbot, questions)
        
        # Affichage des résultats
        afficher_resultats(stats, temps_demarrage)
        
        # Affichage des statistiques du chatbot
        print("\n📊 STATISTIQUES DU CHATBOT :")
        chatbot.afficher_statistiques()
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")
        return 1
    


if __name__ == "__main__":
    exit(main())