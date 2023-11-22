import pandas as pd
import streamlit as st
import numpy as np
import pickle
from lime.lime_tabular import LimeTabularExplainer
import shap
from PIL import Image
import io
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)
filename = 'rfc_model.sav'

# Initialize session state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
    
st.set_page_config(page_title="Deciphering  of Alzheimer's Disease Prediction with Explanations", layout = 'wide', initial_sidebar_state = 'auto')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Deciphering  of Alzheimer's Disease Prediction with Explanations",help="""
'Deciphering Alzheimer's Disease Prediction with Explanations' is a user-friendly web app that 
employs advanced machine learning interpretation techniques like SHAP and Lime to provide transparent 
insights into Alzheimer's disease predictions. Through visualizations such as Explanation chart,Perturbation Analysis, 
SHAP Values, summary plot and force plot the app enhances understanding and transparency in the decision-making processes 
of predictive models for Alzheimer's disease.""")
tx = """"This prediction app is created based on the Alzheimer's disease dataset

The ML model used for the prediction is Random Forest Classifier"""
st.sidebar.info(tx)

@st.cache_data
def load_data():
    alz_df = pd.read_csv("alz_balanc.csv")
    loaded_model1 = pickle.load(open(filename, 'rb'))
    X1 = alz_df.iloc[:,:-1]
    y1 = alz_df.iloc[:,-1]
    return loaded_model1, X1, y1
loaded_model, X, y = load_data()

age = st.number_input("Age of the person",50,100,step=1, help = "Current age of the individual")
cdr = st.number_input("CDRSB(0-20)",0.0,20.0,step=0.1, help="Clinical Dementia Rating Scale Sum of Boxes")
ads11 = st.number_input('ADAS11 (0 - 75)',0.0,75.0,step=0.01,help="Alzheimer's Disease Assessment Scale (11 items) ")
ads13 = st.number_input("ADAS13 (0 - 85)",0.0,85.0,step=0.01,help="Alzheimer's Disease Assessment Scale (13 items) ")
mmse = st.number_input("MMSE (0 - 30)",0.0,30.0,step=0.01, help="Mini-Mental State Examination Score")
rai = st.number_input("RAVLT_immediate (0 - 75)",0.0,75.0,step=0.01, help = "Rey Auditory Verbal Learning Test - Immediate Recall")
ral = st.number_input("RAVLT_learning (-5 - 15)",-5.0,15.0,step=0.01, help="Rey Auditory Verbal Learning Test - Learning Score")
faq = st.number_input("FAQ (0 - 30)",0.0,30.0,step=0.01,help="Functional Activities Questionnaire Score")
hp = st.number_input("Hippocampus (2200,11300)",2200,11300,step=1,help = "Hippocampal volume ")
dic = {'AGE':age, 'CDRSB':cdr, 'ADAS11':ads11, 'ADAS13':ads13, 'MMSE':mmse,
       'RAVLT_immediate':rai, 'RAVLT_learning':ral, 'FAQ':faq, 'Hippocampus':hp}
df = pd.DataFrame([dic])
res = loaded_model.predict(df)

class_nam=['DEMENTIA', 'MCI', 'NL']

if st.button('Predict'):
    st.session_state.button_clicked = not st.session_state.button_clicked

    
if st.session_state.button_clicked:
    if res[0] == 'DEMENTIA':
        st.subheader('Predicted Class - DEMENTIA')
    elif res[0] == 'NL':
        st.subheader('Predicted Class - NL- Normal')
    else:
        st.subheader('Predicted Class - MCI - Mild Cognitive Impairment')
        
    st.write(f"Probability of predicting {res[0]} is {max(loaded_model.predict_proba(df)[0])} ")
    sel = st.radio('Choose the Explainability Principle',options=["LIME","SHAP"])

    if sel == "LIME":
        explainer = LimeTabularExplainer(X.values, 
                                         feature_names=X.columns, 
                                         class_names=['DEMENTIA', 'MCI', 'NL'])


        instance_to_explain = df.iloc[0]

        prediction = loaded_model.predict_proba(df)
        predicted_class = class_nam[np.argmax(prediction)]
        explanation = explainer.explain_instance(instance_to_explain.values, 
                                                     loaded_model.predict_proba, 
                                                     num_features=len(X.columns), 
                                                     top_labels=len(explainer.class_names), 
                                                      labels=[predicted_class])
        st.title('Local Interpretable Model-agnostic Explanations (LIME) for Model Prediction',
                 help="LIME is a technique used to explain the predictions of machine learning models locally. It provides insights into how the model's prediction changes for a specific instance by perturbing the input features.")
        #st.subheader(f'Predicted Class: {predicted_class}')

        # Display Lime explanation details

        for label in explanation.available_labels():
            st.subheader(f'LIME Explanation Chart for Class: {class_nam[label]}',
                        help = "chart represents the impact of different features on the model's prediction. Positive weights indicate a feature's contribution towards the predicted class, while negative weights indicate a feature's contribution towards other classes.")
            fig = explanation.as_pyplot_figure(label)
            st.pyplot(fig)

        # Display feature importance
        st.subheader(f'LIME Feature Importance for the predicted class {predicted_class}',
                    help = "Feature importance shows the contribution of each feature to the predicted class.")
        feature_importance = explanation.as_list(label=np.argmax(prediction))
        if len(feature_importance) > 0:
            st.write(pd.DataFrame(feature_importance, columns=['Feature', 'Weight']))
        else:
            st.write("No feature importance data available for the predicted class.")


    
        # Display perturbation analysis
        st.subheader('LIME Perturbation Analysis')
        feature_mapping = {i: feature_name for i, feature_name in enumerate(X.columns)}
        perturbation_data = explanation.as_map()
        for label, weights in perturbation_data.items():
            st.subheader(f'Perturbation Analysis for Class: {class_nam[label]}',
                        help = """Perturbation analysis displays the impact of different feature values on the model's prediction.

    A positive weight suggests that an increase in the feature value contributes positively to the predicted class.
    Higher values of this feature are associated with a higher likelihood of the predicted class.

    A negative weight indicates that an increase in the feature value contributes negatively to the predicted class.
    Lower values of this feature are associated with a higher likelihood of the predicted class.

    The magnitude of the weight reflects the strength of the feature's impact on the model's prediction.
    A larger magnitude indicates a more significant influence.

    Comparing weights across different features offers insights into the relative importance of each feature for the given instance.
    """)
            #df_perturbation = pd.DataFrame(weights, columns=['Feature', 'Weight'])
            #st.write(df_perturbation)
            df_perturbation = pd.DataFrame(weights, columns=['Feature Number', 'Weight'])
            df_perturbation['Feature Name'] = df_perturbation['Feature Number'].map(feature_mapping)
    
            # Display the DataFrame with feature names
            st.write(df_perturbation[['Feature Name', 'Weight']])
    else:

        st.title("SHapley Additive exPlanations (SHAP)",help = """SHAP values aim to fairly distribute the contribution of each feature to the model's prediction across all possible combinations of features. """)
            # SHAP Explanation
        shexplainer = shap.Explainer(loaded_model)
        shap_values = shexplainer.shap_values(df)


        # Summary Plot
        st.subheader('Summary Plot', help = """The SHAP summary plot provides an overview of feature importance for each class.
        It displays the magnitude and direction of the impact of each feature on the model's output.
        Features with positive SHAP values contribute positively to the predicted class and negative SHAP values contribute 
        negatively to the predicted class.""")
        fig_summary = shap.summary_plot(shap_values, df, show=False,class_names=class_nam)
        st.pyplot(fig_summary)



        # Force Plot
        st.subheader('Force Plot',help = """Force plot shows how each feature contributes to the prediction of the 
        selected class for a given instance. Features with positive contributions increase the likelihood of the 
        predicted class and features with negative contributions decrease the likelihood of the predicted class.
    Red bars represent higher feature values, while blue bars represent lower feature values.""")
        i=0
        for f in class_nam:
            st.write(f"Features contributed for the class {f} ")
            fp = shap.plots.force(shexplainer.expected_value[i], shap_values[i], df, matplotlib = True)
            st.pyplot(fp)
            i+=1

        # Decision Plot
        st.subheader("Decision Plot",help = """Decision plot shows the decision path of the model for a given instance, highlighting the contributions of different features.

    Feature Contributions: Observe how different features influence the decision at various points.

    Positive and Negative Contributions: Positive contributions increase the likelihood of the predicted class, while negative contributions decrease it.""")
        dp = shap.decision_plot(shexplainer.expected_value[1], shap_values[1], df.columns)
        st.pyplot(dp)

