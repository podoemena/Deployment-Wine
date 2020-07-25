import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple  Wine Prediction App
This app predicts the **Wine** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Alcohol = st.sidebar.slider('Alcohol', 11.0,14.8,13.0)
    Malic_acid = st.sidebar.slider('Malic acid', 0.74, 5.8, 2.34)
    Ash = st.sidebar.slider('Ash', 1.36,3.23,2.36)
    Alcalinity_of_Ash= st.sidebar.slider('Alcalinity of Ash', 10.6,30.0,19.5)
    Magnesium = st.sidebar.slider('Magnesium', 70.0,162.0,99.7)
    Total_Phenols = st.sidebar.slider('Total_Phenols', 0.98,3.88,2.29)
    Flavanoids = st.sidebar.slider('Flavanoids', 0.34,5.08,2.03)
    Nonflavanoid_Phenols = st.sidebar.slider('Nonflavanoid_Phenols', 0.13,0.66,0.36)
    Proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58,1.59)
    Colour_Intensity = st.sidebar.slider('Colour Intensity', 1.3,13.0,5.1)
    Hue = st.sidebar.slider('Hue', 0.48,1.71,0.96)
    OD280_OD315_of_diluted_wines = st.sidebar.slider('OD280_OD315_of_diluted_wines', 1.27,4.00,2.61)
    Proline = st.sidebar.slider('Proline', 278,1680,746)
    data = {'Alcohol': Alcohol,
            'Malic_acid': Malic_acid,
            'Ash': Ash,
            'Alcalinity_of_Ash': Alcalinity_of_Ash,
            'Magnesium': Magnesium,
            'Total_Phenols': Total_Phenols,
            'Flavanoids': Flavanoids,
            'Nonflavanoid_Phenols': Nonflavanoid_Phenols,
            'Proanthocyanins': Proanthocyanins,
            'Colour_Intensity': Colour_Intensity,
            'Hue': Hue,
            'OD280_OD315_of_diluted_wines': OD280_OD315_of_diluted_wines,
            'Proline': Proline}
         
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

wine = datasets.load_wine()
X = wine.data
Y = wine.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(wine.target_names)

st.subheader('Prediction')
st.write(wine.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)