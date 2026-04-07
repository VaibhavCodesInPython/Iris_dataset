import streamlit as st
import numpy as np
import pickle

with open('iris_flower_dataset.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Prediction")

sepal_length = st.slider("Sepal Length(cm)",4.0,8.0)
sepal_width = st.slider("Sepal Width(cm)",3.0,8.0)
petal_length = st.slider("Petal Length(cm)",0.5,6.0)
petal_width = st.slider("Petal Width(cm)",0.0,6.0)


if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predict = model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Prediction went successfully! :{species[predict[0]]}")