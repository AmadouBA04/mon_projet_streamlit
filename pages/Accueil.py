import streamlit as st
import base64

# Fonction pour définir l'image d'arrière-plan
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    css_code = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# Définir l'image d'arrière-plan
set_background("ML.jpg")  # Assurez-vous que l'image est dans le bon dossier

# Contenu de la page d'accueil
st.title("🏠 Accueil")
st.write("Bienvenue dans l'application de Machine Learning pour l'étude pronostique !")

st.markdown(
    """
    ### 🌟 Objectif :
    Cette application permet d'analyser les données médicales et de modéliser la probabilité de décès après un AVC.
    
    **Navigation :**
    - 📊 **Données** : Explorer les statistiques et visualisations.
    - 🤖 **Modélisation** : Entraîner des modèles de Machine Learning et Deep Learning.
    - 🔮 **Prédiction** : Faire des prédictions en temps réel.

    

     **NB** : Dans cette application nous utilisons des méthodes de statistique standard, et d’Intelligence artificielle, Présenter les modèles et Justifier leur choix, Interpréter les Résultats.
    """,
    unsafe_allow_html=True
)
