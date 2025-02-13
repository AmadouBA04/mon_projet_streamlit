import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from lifelines import KaplanMeierFitter
import statsmodels.api as sm


# Configuration de la page
st.set_page_config(page_title="Modélisation", page_icon="🧠", layout="wide")

@st.cache_data
def load_data():
    """Charge les données depuis un fichier Excel sans modifier le temps de suivi."""
    file_path = "Donnnées_Projet_M2SID2023_2024_préparées.xlsx"
    df = pd.read_excel(file_path, engine="openpyxl")
    return df

# Charger les données
df = load_data()

st.title("🧠 Modelisation")
st.write("Chargement et préparation des données : Les données sont chargées, la variable cible Evolution est définie, et les variables catégoriques sont encodées.")
st.write("Division des données : Les données sont séparées en ensembles d'entraînement (80%) et de test (20%).")
st.write("Sélection et paramétrage du modèle : L'utilisateur choisit entre une régression logistique, un SVM ou une forêt aléatoire, avec des hyperparamètres ajustables.")
st.write("Entraînement et évaluation : Le modèle est entraîné, puis évalué avec des métriques comme précision, rappel, F1-score, matrice de confusion et Kappa de Cohen.")
st.write("La méthode de Kaplan-Meier estime et affiche les courbes de survie et de décès en fonction d'une variable explicative catégorielle, permettant de comparer la probabilité de survie entre différents groupes.")

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

# Convertir toutes les valeurs en chaînes de caractères (évite l'erreur int/str mixés)
X[categorical_cols] = X[categorical_cols].astype(str)

# Encodage des variables catégoriques avec OneHotEncoder
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), 
                         columns=encoder.get_feature_names_out(), 
                         index=X.index)

# Fusionner avec les autres colonnes numériques
X_final = pd.concat([X_encoded, X.select_dtypes(exclude=["object"])], axis=1)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Barre latérale pour choisir le modèle
st.sidebar.header("Paramètres du modèle")
model_choice = st.sidebar.selectbox("Choisissez un modèle", ["Régression Logistique", "SVM", "Forêt Aléatoire"])

if model_choice == "Régression Logistique":
    # Ajouter une constante pour l'interception
    X_train_sm = sm.add_constant(X_train)

    # Ajuster le modèle avec statsmodels
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()

    # Extraire les coefficients et calculer l'Odds Ratio
    odds_ratios = np.exp(result.params)

    # Extraire les p-values
    p_values = result.pvalues

    # Créer un tableau des résultats
    or_df = pd.DataFrame({
        "Variable": odds_ratios.index,
        "Odds Ratio": odds_ratios.values,
        "p-value": p_values.values
    })

    # Afficher les résultats triés par Odds Ratio
    st.write("### Rapport de cote (Odds Ratio) avec p-value")
    st.write(or_df.sort_values(by="Odds Ratio", ascending=False))


    c_value = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=c_value)
elif model_choice == "SVM":
    kernel_type = st.sidebar.selectbox("Type de noyau", ["linear", "rbf", "poly", "sigmoid"])
    model = SVC(kernel=kernel_type)
else:
    n_estimators = st.sidebar.slider("Nombre d'arbres", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators)

# Entraînement et évaluation

if st.sidebar.button("Entraîner le modèle"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Classes réelles dans y_test :", y_test.value_counts())
    st.write("Prédictions du modèle :", pd.Series(y_pred).value_counts())

    st.write("### Évaluation du modèle")
    st.write(f"Précision : {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Score de précision : {precision_score(y_test, y_pred, average='weighted'):.2f}")
    st.write(f"Score de rappel : {recall_score(y_test, y_pred, average='weighted'):.2f}")
    st.write("Matrice de confusion :")
    st.write(confusion_matrix(y_test, y_pred))

    # Calcul du coefficient de Kappa
    kappa = cohen_kappa_score(y_test, y_pred)

    # Affichage du résultat
    st.write(f"**Coefficient de Kappa de Cohen :** {kappa:.2f}")

# Kaplan-Meier 
st.sidebar.header("Kaplan-Meier")
variable_survie = "Nbj_Temps_Suivi_Après_Traitement"

# Sélection des variables catégoriques
variables_categoriques = df.select_dtypes(include=["object", "category"]).columns.tolist()
variable_exp = st.sidebar.selectbox("Choisir une variable explicative", variables_categoriques)

if st.sidebar.button("Afficher Kaplan-Meier"):
    if variable_survie in df.columns and variable_exp in df.columns:
        
        # Définir les variables de suivi et d'événement
        T = df[variable_survie]  
        E = df["Evolution"]  

        # Vérifier qu'il y a bien plusieurs groupes
        valeurs_uniques = df[variable_exp].unique()
        if len(valeurs_uniques) > 1:
            kmf = KaplanMeierFitter()
            plt.figure(figsize=(10, 5))

            for val in valeurs_uniques:
                # Ajustement de la courbe de survie
                kmf.fit(T[df[variable_exp] == val], event_observed=E[df[variable_exp] == val], label=f"{variable_exp}={val}")
                
                # Tracer la courbe de survie
                kmf.plot_survival_function(label=f"Survie - {variable_exp}={val}")
                
                # Tracer la courbe de probabilité de décès (1 - survie)
                plt.plot(kmf.survival_function_.index, 1 - kmf.survival_function_.iloc[:, 0], linestyle="dashed", label=f"Décès - {variable_exp}={val}")

            plt.xlabel("Temps de suivi")
            plt.ylabel("Probabilité")
            plt.title("Courbes de Kaplan-Meier (Survie et Décès)")
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning(f"Pas assez de variations dans la variable {variable_exp} pour Kaplan-Meier.")
    else:
        st.error("Les variables sélectionnées ne sont pas valides.")
