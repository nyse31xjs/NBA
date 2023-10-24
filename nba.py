import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
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
    st.write("Maintenant que nous avons analysé les données de manière descriptives. Nous développons"
             " un modèle de machine learning nommé K Neirest Neighbors. Notre objectif ? Prédire si un joueur"
             " peut prétendre à une carrière d'au moins 5 ans en NBA étant donné ses caractéristiques personnelles.")
    st.write("### Choix de la distance")
    distance = st.selectbox('Quelle distance souhaitez-vous utiliser ?', ('Manhattan', 'Euclidienne', 'Minkowski'))
    st.write("La distance choisie peut avoir un grand impact sur l'efficacité du modèle, vous "
             " pouvez modidier le type de distance pour voir les effets sur la performance du modèle.")
    st.write("### Performance du modèle")
    st.write("Métrique d'exactitude :")

    st.write("### Prédiction")


    stl = st.slider('Nombre moyen de ballons piqués ? (par match)', 0.0, 3.0, 1.0, step = 0.2)
    st.write("Valeur choisie :", stl)

    min = st.slider('Temps de jeu moyen ? (en minutes)', 0, 48, 10)
    st.write("Valeur choisie :", min)

    pts = st.slider('Nombre moyen de points marqués ? (par match)', 0.0, 30.0, 10.0, step = 0.2)
    st.write("Valeur choisie :", pts)

    ast = st.slider('Nombre moyen de passes décisives ? (par match)', 0.0, 11.0, 6.0, step = 0.2)
    st.write("Valeur choisie :", ast)

    three = st.slider('Nombre moyen de 3 point réussies? (par match)', 0.0, 2.5, 1.0, step = 0.1)
    st.write("Valeur choisie :", three)

    if st.button('Prédire'):
        st.write('La probabilité que le joueur fasse une carrière de plus de 5 ans est :')

