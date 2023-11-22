import streamlit as st
from PIL import Image
import pandas as pd

# Title Section
st.title("Deciphering of Alzheimer's Disease Prediction with Explanations")

# Hero Section
st.image(Image.open("bg.jpg"), width=800)
st.markdown(
    """
    <p style="text-align: center; font-size: 24px">Harnessing the power of Artificial Intelligence for early detection and transparent explanations.</p>
    """,
    unsafe_allow_html=True
)

# Understanding Alzheimer's Disease Section
st.header("Understanding Alzheimer's Disease")
#st.image(Image.open("brain_image.jpg"), width=400, float_="left")
st.markdown(
    """
    Alzheimer's disease, a progressive neurodegenerative disorder that gradually erodes memory, cognitive functions, and daily activities, impacts millions of individuals worldwide. Early detection is crucial for timely intervention and personalized care.
    """
)
#st.video("alzheimers_disease_video.mp4")

# AI in Prediction of Alzheimer's Disease Section
st.header("AI in Prediction of Alzheimer's Disease")
#st.image(Image.open("neural_network_image.jpg"), width=400, float_="left")
st.markdown(
    """
    The Deciphering of Alzheimer's Disease Prediction with Explanations application harnesses the power of Artificial Intelligence to predict Alzheimer's disease with precision. Using a Random Forest classifier, the app considers vital input features such as 'AGE,' 'CDRSB,' 'ADAS11,' 'ADAS13,' 'MMSE,' 'RAVLT_immediate,' 'RAVLT_learning,' 'FAQ,' and 'Hippocampus,'  to classify individuals into one of three categories: 'DEMENTIA,' 'MCI' (Mild Cognitive Impairment), or 'Normal.'
    """
)

# Transparent Explanations with LIME Section
st.header("Transparent Explanations with LIME")
#st.image(Image.open("lock_image.jpg"), width=400, float_="left")
st.markdown(
    """
    Understanding the rationale behind every prediction is paramount. Our application employs LIME (Local Interpretable Model-agnostic Explanations) to provide clear, local insights into why the model made a specific prediction for an individual. LIME generates easily understandable explanations, allowing users to grasp the influential features and factors contributing to the predicted class.
    """
)

# SHAP Values for Comprehensive Understanding Section
st.header("SHAP Values for Comprehensive Understanding")
#st.image(Image.open("puzzle_image.jpg"), width=400, float_="left")
st.markdown(
    """
    SHAP (SHapley Additive exPlanations) values enhance interpretability by quantifying the impact of each feature on the model's output. Our application utilizes SHAP to create interactive visualizations, including Summary Plots, force plots, and decision plots. This comprehensive approach empowers users to explore the relative importance of different features in predicting each class, fostering a deeper understanding of the model's decision logic.
    """
)

# User-Friendly Input and Insights Section
st.header("User-Friendly Input and Insights")
#st.image(Image.open("interface_mockup.jpg"), width=1000)
st.markdown(
    """
    Inputting and interpreting data is simplified through our user-friendly interface. Users can effortlessly input relevant features, obtain predictions for Alzheimer's disease classes, and delve into detailed explanations.
    """
)

# Call to Action
#st.button("Experience AI-powered Alzheimer's Disease Prediction")
st.markdown(
    """
    <h2 style="text-align: center; font-size: 36px">Experience the power of  interpretable AI in Alzheimer's disease prediction with 'Deciphering Alzheimer's Disease Prediction with Explanations.' </h2>
    """,
    unsafe_allow_html=True
)
