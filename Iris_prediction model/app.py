import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


df = pd.read_csv('iris.csv')
with open('iris_flower_dataset.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris AI", layout="wide")

st.title("Iris Flower AI Predictor + 3D Visualizer")

# Layout (better UI)
col1, col2 = st.columns(2)


st.subheader("Input Features")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width = st.slider("Sepal Width (cm)", 3.0, 8.0)
petal_length = st.slider("Petal Length (cm)", 0.5, 6.0)
petal_width = st.slider("Petal Width (cm)", 0.0, 6.0)



if st.button("Predict"):
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediction
    predict = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    species = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species[predict]

    # Confidence
    confidence = np.max(proba)

    st.success(f"Prediction: **{predicted_species}**")
    st.info(f"Confidence: **{confidence:.2f}**")

    # Color mapping
    color_map = {
        "Setosa": 'blue',
        "Versicolor": 'green',
        "Virginica": 'purple'
    }

    # 2D Plot 
    st.subheader(" 2D Visualization")

    fig, ax = plt.subplots()

    for sp in df['species'].unique():
        subset = df[df['species'] == sp]
        ax.scatter(
            subset['petal_length'],
            subset['petal_width'],
            label=sp,
            alpha=0.5
        )

    ax.scatter(
        petal_length,
        petal_width,
        color=color_map[predicted_species],
        s=120,
        edgecolors='black',
        label=f'Your Input ({predicted_species})'
    )

    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend()

    st.pyplot(fig)

    # 3D Plot (Hologram-like)
    st.subheader("3D Interactive Visualization using Hologram Effect")

    fig3d = px.scatter_3d(
        df,
        x='petal_length',
        y='petal_width',
        z='sepal_length',
        color='species',
        title="3D Iris Dataset"
    )

    # Add user point with bigger hologram effect
    fig3d.add_scatter3d(
        x=[petal_length],
        y=[petal_width],
        z=[sepal_length],
        mode='markers',
        marker=dict(
            size=40,       # Bigger point
            color='black',
            opacity=0.9
        ),
        name=f'Your Input ({predicted_species})'
    )

    # Make the 3D figure larger and more immersive
    fig3d.update_layout(
        width=1200,    # Increase width
        height=900,    # Increase height
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title='Petal Length',
            yaxis_title='Petal Width',
            zaxis_title='Sepal Length',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust viewing angle
            )
        )
    )

    st.plotly_chart(fig3d, use_container_width=True)