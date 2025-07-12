# import json
# import os
# import nlpaug.augmenter.word as naw
# import nltk
# nltk.download("wordnet")

# from utils import charger_donnees_json

# data_path = "data/Nestle-HR-FAQ.json"
# output_path = "data/Nestle-HR-FAQ-augmente.json"

# def augmenter_donnees():
#     questions, reponses = charger_donnees_json(data_path)

#     print("🔁 Génération des variantes synonymiques (x2)...")

#     aug = naw.SynonymAug(aug_src="wordnet")
#     nouvelles_qa = []

#     for q, r in zip(questions, reponses):
#         nouvelles_qa.append({"question": q, "response": r})
#         try:
#             for _ in range(2):  # 2 variantes par question
#                 q_aug = aug.augment(q)
#                 if q_aug and isinstance(q_aug, str) and q_aug != q:
#                     nouvelles_qa.append({"question": q_aug, "response": r})
#         except Exception as e:
#             print(f"Erreur sur : {q} → {e}")

#     print(f"✅ {len(nouvelles_qa)} entrées générées au total.")

#     data_struct = {"faq": {"donnees_augmentees": nouvelles_qa}}

#     os.makedirs("data", exist_ok=True)

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(data_struct, f, ensure_ascii=False, indent=2)

#     print(f"💾 Fichier sauvegardé → {output_path}")

# if __name__ == "__main__":
#     augmenter_donnees()

"""
Augmenteur de données FAQ RH - Génération de variantes synonymiques
Auteur: Système d'augmentation de données NLP
Description: Génère automatiquement des variantes de questions FAQ pour enrichir les données d'entraînement
"""

import json
import os
import nltk
import nlpaug.augmenter.word as naw
from typing import Dict, List, Optional
from utils import charger_donnees_json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def telecharger_ressources_nltk():
    """
    Télécharge les ressources NLTK nécessaires pour l'augmentation de données
    """
    print("📦 Téléchargement des ressources NLTK...")
    
    ressources = [
        "wordnet",
        "averaged_perceptron_tagger", 
        "averaged_perceptron_tagger_eng",
        "punkt"
    ]
    
    for ressource in ressources:
        try:
            nltk.download(ressource, quiet=True)
            print(f"✅ {ressource} téléchargé")
        except Exception as e:
            print(f"⚠️ Échec du téléchargement de {ressource}: {e}")
    
    print("🎯 Ressources NLTK prêtes!\n")


class FAQAugmenter:
    """
    Classe pour augmenter les données FAQ RH avec des variantes synonymiques
    
    Attributes:
        data_path (str): Chemin vers le fichier JSON source
        output_path (str): Chemin vers le fichier JSON de sortie
        augmenter: Instance de l'augmenteur de synonymes
    """
    
    def __init__(self, data_path: str, output_path: str):
        """
        Initialise l'augmenteur FAQ
        
        Args:
            data_path (str): Chemin vers le fichier JSON source
            output_path (str): Chemin vers le fichier JSON de sortie
        """
        self.data_path = data_path
        self.output_path = output_path
        self.augmenter = naw.SynonymAug(aug_src="wordnet")
        self.statistiques = {
            "questions_originales": 0,
            "variantes_generees": 0,
            "erreurs_generation": 0,
            "themes_traites": 0
        }
    
    def lire_donnees_source(self) -> Dict:
        """
        Lit le fichier JSON source avec gestion d'erreurs
        
        Returns:
            dict: Données FAQ chargées depuis le fichier JSON
            
        Raises:
            FileNotFoundError: Si le fichier source n'existe pas
            json.JSONDecodeError: Si le format JSON est invalide
        """
        print("🔁 Lecture du fichier JSON original...")
        
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validation de la structure des données
            if "faq" not in data:
                raise ValueError("Structure JSON invalide : clé 'faq' manquante")
            
            print(f"✅ Fichier lu avec succès : {self.data_path}")
            print(f"📊 {len(data['faq'])} thèmes détectés")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ Erreur : Fichier introuvable - {self.data_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ Erreur : Format JSON invalide - {self.data_path}")
            print(f"   Détail : {e}")
            raise
        except Exception as e:
            print(f"❌ Erreur inattendue lors de la lecture : {e}")
            raise
    
    def generer_variantes_question(self, question: str, nb_variantes: int = 2) -> List[str]:
        """
        Génère des variantes synonymiques d'une question
        
        Args:
            question (str): Question originale à augmenter
            nb_variantes (int): Nombre de variantes à générer (défaut: 2)
            
        Returns:
            List[str]: Liste des variantes générées (filtrées et uniques)
        """
        variantes = []
        
        if not question or not question.strip():
            print("⚠️ Question vide ou invalide détectée")
            return variantes
        
        try:
            tentatives = 0
            max_tentatives = nb_variantes * 2  # Permet plus de tentatives
            
            while len(variantes) < nb_variantes and tentatives < max_tentatives:
                variante = self.augmenter.augment(question)
                
                # Validation de la variante
                if self._valider_variante(variante, question, variantes):
                    variantes.append(variante)
                
                tentatives += 1
                
        except Exception as e:
            print(f"⚠️ Erreur pour '{question[:50]}...' → {e}")
            self.statistiques["erreurs_generation"] += 1
        
        return variantes
    
    def _valider_variante(self, variante: str, question_originale: str, variantes_existantes: List[str]) -> bool:
        """
        Valide qu'une variante est acceptable
        
        Args:
            variante (str): Variante à valider
            question_originale (str): Question originale
            variantes_existantes (List[str]): Variantes déjà générées
            
        Returns:
            bool: True si la variante est valide
        """
        return (
            isinstance(variante, str) and
            variante.strip() and
            variante != question_originale and
            variante not in variantes_existantes and
            len(variante) > 10  # Évite les variantes trop courtes
        )
    
    def augmenter_donnees_par_theme(self) -> Dict:
        """
        Génère des variantes synonymiques pour chaque question dans chaque thème RH
        
        Returns:
            dict: Données FAQ augmentées avec les variantes
        """
        # Charger les données source
        data = self.lire_donnees_source()
        
        # Initialiser la structure des données augmentées
        data_augmente = {"faq": {}}
        
        print("🧠 Génération des variantes par section RH...")
        print("=" * 50)
        
        # Parcourir chaque thème et ses questions
        for theme, items in data["faq"].items():
            print(f"🔹 Thème : {theme}")
            data_augmente["faq"][theme] = []
            
            questions_theme = 0
            
            for item in items:
                question = item.get("question", "")
                reponse = item.get("response", "")
                
                if not question:
                    print(f"⚠️ Question vide détectée dans le thème {theme}")
                    continue
                
                # Ajouter la question originale
                data_augmente["faq"][theme].append({
                    "question": question,
                    "response": reponse
                })
                questions_theme += 1
                self.statistiques["questions_originales"] += 1
                
                # Générer et ajouter les variantes
                variantes = self.generer_variantes_question(question, nb_variantes=2)
                
                for variante in variantes:
                    data_augmente["faq"][theme].append({
                        "question": variante,
                        "response": reponse
                    })
                    questions_theme += 1
                    self.statistiques["variantes_generees"] += 1
            
            print(f"   ✅ {questions_theme} questions générées pour ce thème")
            self.statistiques["themes_traites"] += 1
        
        print("=" * 50)
        total_questions = sum(len(items) for items in data_augmente["faq"].values())
        print(f"🎯 {total_questions} questions générées au total")
        
        return data_augmente
    
    def sauvegarder_donnees(self, data_augmente: Dict):
        """
        Sauvegarde les données augmentées dans un fichier JSON
        
        Args:
            data_augmente (dict): Données FAQ augmentées
            
        Raises:
            Exception: Si la sauvegarde échoue
        """
        print("💾 Sauvegarde des données augmentées...")
        
        # Créer le répertoire de sortie si nécessaire
        repertoire_sortie = os.path.dirname(self.output_path)
        if repertoire_sortie:
            os.makedirs(repertoire_sortie, exist_ok=True)
        
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(data_augmente, f, indent=2, ensure_ascii=False)
            
            # Vérifier la taille du fichier
            taille_fichier = os.path.getsize(self.output_path) / 1024  # en KB
            print(f"✅ Fichier sauvegardé : {self.output_path}")
            print(f"📏 Taille du fichier : {taille_fichier:.1f} KB")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde : {e}")
            raise
    
    def afficher_statistiques(self):
        """
        Affiche les statistiques détaillées de l'augmentation
        """
        print("\n" + "="*60)
        print("📊 STATISTIQUES DÉTAILLÉES")
        print("="*60)
        print(f"🔹 Questions originales    : {self.statistiques['questions_originales']}")
        print(f"🔹 Variantes générées      : {self.statistiques['variantes_generees']}")
        print(f"🔹 Total questions         : {self.statistiques['questions_originales'] + self.statistiques['variantes_generees']}")
        print(f"🔹 Thèmes traités          : {self.statistiques['themes_traites']}")
        print(f"🔹 Erreurs de génération   : {self.statistiques['erreurs_generation']}")
        
        if self.statistiques['questions_originales'] > 0:
            taux_augmentation = (self.statistiques['variantes_generees'] / 
                               self.statistiques['questions_originales']) * 100
            print(f"🔹 Taux d'augmentation     : {taux_augmentation:.1f}%")
        
        print("="*60)
    
    def executer_augmentation(self):
        """
        Exécute le processus complet d'augmentation des données FAQ
        """
        print("🚀 Démarrage de l'augmentation des données FAQ RH...")
        print("="*60)
        
        try:
            # Étape 1: Générer les données augmentées
            data_augmente = self.augmenter_donnees_par_theme()
            
            # Étape 2: Sauvegarder les résultats
            self.sauvegarder_donnees(data_augmente)
            
            # Étape 3: Afficher les statistiques
            self.afficher_statistiques()
            
            print(f"\n🎯 PROCESSUS TERMINÉ AVEC SUCCÈS!")
            print(f"📄 Fichier de sortie : {self.output_path}")
            
        except Exception as e:
            print(f"❌ Échec de l'augmentation : {e}")
            print("🔧 Vérifiez les chemins de fichiers et les permissions")
            raise


def main():
    """
    Fonction principale pour exécuter l'augmentation des données FAQ
    """
    # Configuration des chemins
    data_path = "data/Nestle-HR-FAQ.json"
    output_path = "data/Nestle-HR-FAQ-augmente.json"
    
    # Télécharger les ressources NLTK
    telecharger_ressources_nltk()
    
    # Créer et exécuter l'augmenteur
    try:
        augmenteur = FAQAugmenter(data_path, output_path)
        augmenteur.executer_augmentation()
        
    except KeyboardInterrupt:
        print("\n⚠️ Processus interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        print("🔧 Vérifiez votre configuration et réessayez")


if __name__ == "__main__":
    main()