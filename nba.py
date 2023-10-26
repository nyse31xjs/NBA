import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('nba-players.csv')

df = df.drop('Unnamed: 0', axis = 1)
#remove 1st column

data = df.copy()
#data is a copy of df (df is not changed anymore)

data.drop_duplicates(subset=['name'], keep='first', inplace=True)
#keep first is arbitrary

sns.pairplot(data[['gp','pts','3p','target_5yrs']], hue = 'target_5yrs')


st.sidebar.title("Sommaire")
pages = ["Contexte","Exploration des données","Machine learning"]
page = st.sidebar.radio("Aller vers la page...", pages)

if page == pages[0]:
    st.write("# Contexte du projet")
    st.write("Le monde du sport professionnel demande une rigueur extreme.")
    st.image("logo.jpg")

    st.write("## Description des variables")


elif page == pages[1]:
    st.write("# Exploration des données")
    st.image("stat.jpg")
    st.write("## La base de données")
    st.write("Nombre d'individus dans la base données :")
    st.write(df.shape[0])
    st.write("Nombre de variables dans la base données :")
    st.write(df.shape[1])

    if st.checkbox("Afficher le nombre de valeurs manquantes dans la base de données :"):
        st.dataframe(df.isna().sum())

    if st.checkbox("Afficher le nombre de doublons dans la base de données :"):
        st.dataframe(df.duplicated().sum())

    st.write("## Analyse de données")
    st.write("Nous avons retiré les doublons.")

    st.write("## Matrice de visualisation")
    variables = st.multiselect('Quelle variable voulez-vous visualiser ?', data.columns.tolist(), ['min', 'pts'])
    st.pyplot(sns.pairplot(data[variables]), hue = "target_5yrs")

    st.write("## Couple de variables interactif")
    variable1 = st.selectbox('Première variable', tuple(data.columns))
    variable2 = st.selectbox('Deuxième variable', tuple(data.columns) )
    st.plotly_chart(px.scatter(df, x=variable1, y=variable2, color = "target_5yrs", marginal_y="violin",
           marginal_x="box"))

    st.write("## Matrice de corrélation")
    st.plotly_chart(px.imshow(data.corr(), color_continuous_scale=px.colors.sequential.Cividis_r))

elif page == pages[2]:
    st.write("# Machine learning")
    st.image("ml.jpg")
    st.write("Maintenant que nous avons analysé les données de manière descriptives. Nous développons"
             " un modèle de machine learning qui est une regression logistique. Notre objectif ? Prédire si un joueur"
             " peut prétendre à une carrière d'au moins 5 ans en NBA étant donné ses caractéristiques de jeu.")
    st.write("### Procédure")
    st.write("Pour développer ce modèle, nous avons passer par plusieurs étapes. Nous avons du régler"
             " le problème du jeu de données déséquilibré : effectivement dans notre dataset original"
             " les joueurs avec une carrière de plus de 5 ans étaient beaucoup plus nombreux, alors pour ne pas biaiser"
             " l'apprentissage du modèle, nous avons réduit le nombre de ligne en créant un nouveau dataset équilibré"
             " (Undersamping).")
    st.write("Nous avons standardisé les données pour ne pas favoriser Ensuite nous avons procédé à la sélection des variables importantes pour le modèle, nous avons "
             " utilisé la méthode RFE (recursive feature elimination), le modèle est estimé avec toutes les variables, "
             " puis tour à tour, la variable ayant le moins d'impact sur le modèle est eliminée. Nous avons décidé arbitrairement de garder les "
             " 6 variables les plus importantes. Finalement nous en retenons 5 car deux des 6 variables retenues par RFE sont linéairement "
             " positivement corrélées.")

    st.write("### Prédiction")

    st.write("Rentrez les caractéristiques du joueur.")

    gp = st.slider('Nombre de matchs joués ?', 5, 100, 30, step = 1)
    st.write("Valeur choisie :", gp)

    fga = st.slider('Nombre moyen de tentatives par match ?', 0.0, 25.0, 0.1)
    st.write("Valeur choisie :", fga)

    fg = st.slider('Pourcentage moyen de paniers réalisés par match ?', 0, 100, 10, step = 1)
    st.write("Valeur choisie :", fg)

    threepa = st.slider('Nombre moyen de tentatives de panier à trois points par match ?', 0.0, 10.0, 6.0, step = 0.2)
    st.write("Valeur choisie :", threepa)

    oreb = st.slider('Nombre moyen de rebonds offensifs par match ?', 0.0, 6.0, 1.0, step = 0.1)
    st.write("Valeur choisie :", oreb)

    data = np.array([(gp-58.6)/17.6, (fga-5.61)/3.4, (fg-43.8)/6.25, (threepa-0.77)/1.03, (oreb-0.95)/0.74])

    if st.button('Prédire'):
        #load le modele
        loaded_model = joblib.load('logistic.joblib')

        st.write('Prédiction du modèle :')
        if loaded_model.predict(data.reshape(1, -1)) == 0 :
            st.write('Le joueur ne fera probablement pas une carrière d\'au moins 5ans')
        elif loaded_model.predict(data.reshape(1, -1)) == 1 :
            st.write('Le joueur fera probablement une carrière d\'au moins 5ans')
