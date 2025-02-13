import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# Charger les données
df = pd.read_excel("Donnnées_Projet_M2SID2023_2024_préparées.xlsx")

st.title("📊 Données et Statistiques")
st.write("Il s’agit de données de patients atteints d’accident cérébral vasculaire (AVC), traités et suivis.")
st.write("NB : Ces données ont été produites juste pour cette expérience, aucune réutilisation ne sera reconnue !")

# 📌 Définition des variables numériques et catégorielles dès le début
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# 📌 Barre latérale : Options d'affichage
st.sidebar.subheader("AFFICHAGE")
show_data = st.sidebar.checkbox("DONNÉES")
show_stats = st.sidebar.checkbox("STATISTIQUES ET ANALYSES")

if show_data:
    st.write("### Aperçu des données :")
    st.write(df.head())
    st.write("#### Liste des variables :", df.columns.tolist())

if show_stats:
    analysis_type = st.sidebar.radio("Choisissez le type d'analyse :", ["Analyse univariée", "Analyse bivariée"])

    # 🔹 ANALYSE UNIVARIÉE
    if analysis_type == "Analyse univariée":
        st.write("## Analyse univariée")

        # 📌 Statistiques pour les variables numériques
        # 📌 Statistiques descriptives de AGE
        if "AGE" in df.columns:
            st.write("### Statistiques descriptives de AGE")
            st.write(df["AGE"].describe())
        else:
            st.write("La variable 'AGE' n'est pas présente dans les données.")

        # Histogramme
        # 📌 Vérifier si 'AGE' est bien dans les colonnes
        if "AGE" in df.columns:
            st.write("### Histogramme de l'âge")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["AGE"].dropna(), bins=20, kde=True, color="skyblue", edgecolor="black", ax=ax)
            st.pyplot(fig)
        else:
            st.write("⚠️ La variable 'AGE' n'est pas présente dans les données.")

        # 📌 Statistiques pour les variables catégorielles
        for col in categorical_columns:
            st.write(f"### Répartition de {col}")
            fig, ax = plt.subplots(figsize=(5, 5))
            df[col].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax, cmap="coolwarm")
            ax.set_ylabel("")
            st.pyplot(fig)

    # 🔹 ANALYSE BIVARIÉE
    elif analysis_type == "Analyse bivariée":
        st.write("## Analyse bivariée")

        # 📌 Relations entre variables numériques et Evolution
        st.write("### Relation entre variables numériques et Evolution")
        for col in numeric_columns:
            if col != "AGE":  # AGE est traité séparément
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x="Evolution", y=col, data=df, ax=ax)
                st.pyplot(fig)

        # 📌 Relation AGE et Evolution
        st.write("### Relation entre AGE et Evolution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Evolution", y="AGE", data=df, ax=ax)
        st.pyplot(fig)

        # 📌 Relations entre variables catégorielles et Evolution
        st.write("### Relations entre variables catégorielles et Evolution")
        for col in categorical_columns:
            cross_tab = pd.crosstab(df[col], df["Evolution"])
            st.write(f"#### {col} vs Evolution")
            st.write(cross_tab)
            fig, ax = plt.subplots(figsize=(5, 5))
            cross_tab.plot(kind="bar", stacked=True, colormap="coolwarm", ax=ax)
            st.pyplot(fig)

        # 🔹 Matrice de corrélation finale avec variables spécifiques
        st.write("## Matrice de corrélation entre variables clés")
        selected_columns = ["AGE", "Evolution", "Nbj_Premiers_Signe_Adm_Hop", 
                            "Nbj_Adm_Hop_Prise_Charge", "Diabete", "SEXE"]
        
        # Vérifier si toutes les colonnes sélectionnées existent
        selected_columns = ["AGE", "Evolution", "Nbj_Premiers_Signe_Adm_Hop", "Diabete", "SEXE"]

        # Conversion des variables catégorielles en numériques
        df_encoded = pd.get_dummies(df[selected_columns], drop_first=True)

        corr_matrix = df_encoded.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)
