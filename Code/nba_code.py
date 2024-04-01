import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.express as px
import joblib

#database nettoyée depuis le notebook
data = pd.read_csv('df_nba.csv')

st.sidebar.title("Sommaire")
pages = ["Contexte","Exploration des données","Machine learning"]
page = st.sidebar.radio("Aller vers la page...", pages)

if page == pages[0]:
    st.write("# Contexte du projet")
    st.write("Dans l'univers compétitif de la NBA, la capacité à anticiper "
             "la longévité d'une carrière sportive est devenue une quête cruciale. Les équipes de gestion et "
             "les athlètes cherchent inlassablement des moyens d'optimiser les décisions liées au recrutement, "
             "aux contrats et à la gestion de carrière. C'est dans ce contexte que l'application de techniques "
             "de machine learning devient un outil novateur, offrant la promesse d'apporter des éclairages inédits.")
    st.image("logo.jpg")

    st.write("## Description des variables (en anglais)")
    st.write("name Name of NBA player")
    st.write("gp Number of games played")
    st.write("min Number of minutes played per game")
    st.write("pts Average number of points per game")
    st.write("fgm Average number of field goals made per game")
    st.write("fga Average number of field goal attempts per game")
    st.write("fg Average percent of field goals made per game")
    st.write("3p_made Average number of three-point field goals made per game")
    st.write("3pa Average number of three-point field goal attempts per game")
    st.write("3p Average percent of three-point field goals made per game")
    st.write("ftm Average number of free throws made per game")
    st.write("fta Average number of free throw attempts per game")
    st.write("ft Average percent of free throws made per game")
    st.write("oreb Average number of offensive rebounds per game")
    st.write("dreb Average number of defensive rebounds per game")
    st.write("reb Average number of rebounds per game")
    st.write("ast Average number of assists per game")
    st.write("stl Average number of steals per game")
    st.write("blk Average number of blocks per game")
    st.write("tov Average number of turnovers per game")
    st.write("target_5yrs 1 if career duration >= 5 yrs, 0 otherwise")

    st.write("## Source des données")
    st.write("Kaggle")

elif page == pages[1]:
    st.write("# Exploration des données")
    st.image("stat.jpg")
    st.write("## La base de données")
    st.write("Nombre d'individus dans la base données :")
    st.write(data.shape[0])
    st.write("Nombre de variables dans la base données :")
    st.write(data.shape[1])

    if st.checkbox("Afficher le nombre de valeurs manquantes dans la base de données :"):
        st.dataframe(data.isna().sum())

    if st.checkbox("Afficher le nombre de doublons dans la base de données :"):
        st.dataframe(data.duplicated().sum())

    st.write("## Analyse de données")
    st.write("Nous avons retiré les doublons.")

    st.write("## Matrice de visualisation")
    variables = st.multiselect('Quelle variable voulez-vous visualiser ?', data.columns.tolist(), ['min', 'pts'])
    st.pyplot(sns.pairplot(data[variables]), hue = "target_5yrs")

    st.write("## Couple de variables interactif")
    variable1 = st.selectbox('Première variable', tuple(data.columns))
    variable2 = st.selectbox('Deuxième variable', tuple(data.columns) )
    st.plotly_chart(px.scatter(data, x=variable1, y=variable2, color = "target_5yrs", marginal_y="violin",
           marginal_x="box"))

    st.write("## Matrice de corrélation")
    st.plotly_chart(px.imshow(data.corr(), color_continuous_scale=px.colors.sequential.Cividis_r))

elif page == pages[2]:
    st.write("# Machine learning")
    st.image("ml.jpg")
    st.write("Maintenant que nous avons analysé les données de manière descriptive. Nous développons"
             " un modèle de machine learning qui est une regression logistique. Notre objectif ? Prédire si un joueur"
             " peut prétendre à une carrière d'au moins 5 ans en NBA étant donné ses caractéristiques de jeu.")
    st.write("### Procédure")
    st.write("Pour développer ce modèle, nous sommes passés par plusieurs étapes. Nous avons du régler"
             " le problème du jeu de données déséquilibré : effectivement dans notre dataset original"
             " les joueurs avec une carrière de plus de 5 ans étaient beaucoup plus nombreux, alors pour ne pas biaiser"
             " l'apprentissage du modèle, nous avons réduit le nombre de ligne en créant un nouveau dataset équilibré"
             " (Undersamping).")
    st.write("Nous standardisons les données pour ne pas favoriser l'impact d'une variable par rapport à une autre sur le modèle. "
             "Ensuite nous avons procédé à la sélection des variables importantes pour le modèle, nous avons "
             " utilisé la méthode RFE (recursive feature elimination), le modèle est estimé avec toutes les variables, "
             " puis tour à tour, la variable ayant le moins d'impact sur le modèle est eliminée. Nous avons décidé de garder les "
             " 3 variables les plus importantes, étant donné que les variables supplémentaires n'enrichissent que très peu "
             "le modèle. Finalement nous en retenons 2 car deux des 3 variables retenues par RFE sont linéairement "
             " positivement très corrélées.")

    st.write("### Prédiction")

    st.write("Rentrez les caractéristiques du joueur.")

    gp = st.slider('Nombre de matchs joués ?', 5, 100, 30, step = 1)
    st.write("Valeur choisie :", gp)

    threepa_made = st.slider('Nombre de paniers à 3 points moyen ?', 0.0, 5.0, 0.1)
    st.write("Valeur choisie :", threepa_made)

    if st.button('Prédire'):
        # On importe la pipeline
        logit_model = joblib.load('logit_model.joblib')
        # Données input
        new_data = np.array([[gp, threepa_made]])
        st.write('Prédiction du modèle :')
        if logit_model.predict(new_data) == 0 :
            st.write('Le joueur ne fera probablement pas une carrière d\'au moins 5ans')
        elif logit_model.predict(new_data) == 1 :
            st.write('Le joueur fera probablement une carrière d\'au moins 5ans')

    st.write("On peut s'intérroger sur la pertinence de la variable 'nombre de matchs joués', est-ce que la variable "
             "cible n'a pas un impact sur le nombre de matchs joués ? Ainsi nous aurions probablement affaire à un problème "
             "d'endogénéité. ")
