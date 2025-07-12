"""
spellchecker.py
----------------
Module pour prétraiter les questions utilisateur :
- Nettoyage (minuscules, retrait ponctuation)
- Correction orthographique basique
"""

import re
from textblob import TextBlob

def nettoyer_question(texte: str) -> str:
    """
    Met en minuscules et enlève la ponctuation inutile.
    
    Args:
        texte (str): Texte brut
        
    Returns:
        str: Texte nettoyé
    """
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", "", texte)
    return texte

def corriger_orthographe(texte: str) -> str:
    """
    Corrige les fautes d'orthographe simples avec TextBlob.
    
    Args:
        texte (str): Texte nettoyé
        
    Returns:
        str: Texte corrigé
    """
    try:
        blob = TextBlob(texte)
        return str(blob.correct())
    except Exception as e:
        print(f"⚠️ Erreur de correction orthographique : {e}")
        return texte

def preprocess_question(texte: str) -> str:
    """
    Pipeline complet : nettoyage + correction.
    
    Args:
        texte (str): Texte brut de l'utilisateur
        
    Returns:
        str: Texte prêt pour le modèle
    """
    nettoye = nettoyer_question(texte)
    corrige = corriger_orthographe(nettoye)
    return corrige
