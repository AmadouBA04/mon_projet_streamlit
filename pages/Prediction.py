import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Configuration de la page
st.set_page_config(page_title="Prédiction", page_icon="🔮", layout="wide")

@st.cache_data
def load_data():
    """Charge les données depuis un fichier Excel sans modifier le temps de suivi."""
    file_path = "Donnnées_Projet_M2SID2023_2024_préparées.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")
    return df

# Charger les données
df = load_data()

st.title("🔮 Prédiction")

# Vérifier si la colonne Evolution existe et la convertir
if "Evolution" in df.columns:
    if df["Evolution"].dtype == "object":
        df["Evolution"] = df["Evolution"].map({"Vivant": 1, "Deces": 0})
else:
    df["Evolution"] = pd.Series(dtype="float64")

# Séparer les features et la cible
X = df.drop(columns=["Evolution"], errors="ignore")
y = df["Evolution"]

# Sélectionner uniquement les colonnes catégoriques
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].astype(str)

# Encodage des variables catégoriques
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), 
                         columns=encoder.get_feature_names_out(), 
                         index=X.index)

# Fusionner avec les autres colonnes numériques
X_final = pd.concat([X_encoded, X.select_dtypes(exclude=["object"])], axis=1)

# Reconstruire les modèles avec de nouveaux noms
logistic_model_pred = LogisticRegression()
svm_model_pred = SVC()
rf_model_pred = RandomForestClassifier()

# Entraîner les nouveaux modèles
logistic_model_pred.fit(X_final, y)
svm_model_pred.fit(X_final, y)
rf_model_pred.fit(X_final, y)

# Sauvegarder les modèles pour une utilisation ultérieure
joblib.dump(logistic_model_pred, "logistic_model_pred.pkl")
joblib.dump(svm_model_pred, "svm_model_pred.pkl")
joblib.dump(rf_model_pred, "rf_model_pred.pkl")

# Sélection du modèle pour la prédiction
st.sidebar.header("Choix du modèle")
model_choice = st.sidebar.selectbox("Sélectionner un modèle", 
                                    ["Régression Logistique", "SVM", "Forêt Aléatoire"])

# Saisie utilisateur pour les nouvelles prédictions
st.write("### Entrez les caractéristiques du patient pour la prédiction")
user_input = {}

for col in X.columns:
    if col in categorical_cols:
        user_input[col] = st.selectbox(f"{col}", df[col].unique())
    else:
        user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

# Transformer l'entrée utilisateur
user_df = pd.DataFrame([user_input])
user_df[categorical_cols] = user_df[categorical_cols].astype(str)
user_encoded = pd.DataFrame(encoder.transform(user_df[categorical_cols]), 
                            columns=encoder.get_feature_names_out())
user_final = pd.concat([user_encoded, user_df.select_dtypes(exclude=["object"])], axis=1)

# Charger le bon modèle et faire la prédiction
if st.sidebar.button("Prédire"):
    if model_choice == "Régression Logistique":
        model = joblib.load("logistic_model_pred.pkl")
    elif model_choice == "SVM":
        model = joblib.load("svm_model_pred.pkl")
    else:
        model = joblib.load("rf_model_pred.pkl")
    
    prediction = model.predict(user_final)
    resultat = "Vivant" if prediction[0] == 1 else "Décès"
    st.write(f"### Résultat de la prédiction : {resultat}")
