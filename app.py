import streamlit as st
# Définir la configuration de la page

st.set_page_config(page_title="APPLICATION_BIOSTATISTIQUE", page_icon="📊", layout="wide")




# Appliquer du CSS pour personnaliser la page
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
set_background("AC.jpeg")  # Assurez-vous que l'image est dans le bon dossier
st.write("# Bienvenue dans l'application de Machine Learning pour l'étude pronostique complète de la survenue de décès après le traitement !")


# Contenu principal
st.sidebar.success("Sélectionnez une page ci-dessus.")
