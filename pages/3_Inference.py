import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
# saving and loading model
from joblib import load, dump
# Local Interpretable Model-Agnostic Explanations for machine learning classifiers
from lime import lime_tabular

from utils import load_data
from utils import plot_confusion_matrix, plot_feature_importance, get_classification_report


df, X, y = load_data()
feature_names = X.columns
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Model
rf = load('./saved_model/rf_model')
# prediction
y_pred = rf.predict(x_test)



# Dash board
st.title(':violet[Wine] Type Prediction')
st.markdown('Predict wine type using the ingredients values')


tab1, tab2, tab3 = st.tabs(['Test Data :clipboard:', 'Model Performance ✨', 'Inference 🌟'])

with tab1:
    st.header('Test Data')
    st.write(x_test)


with tab2:

    col_1, col_2 = st.columns(2)
    with col_1:
        conf_mat_fig = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(fig=conf_mat_fig, use_container_width=True)
    with col_2:
        feature_importance_fig = plot_feature_importance(rf, feature_names)
        st.pyplot(fig=feature_importance_fig, use_container_width=True)
    
    
    # horizontal line
    st.divider()
    st.header('Classification Report 📝')
    classification_report = get_classification_report(y_test, y_pred)
    st.code(classification_report)



with tab3:
    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown('Model Inputs')
        sliders = []
        for col in X.columns:
            slider = st.slider(
                label=col, 
                min_value=float(np.min(df[col])), 
                max_value=float(np.max(df[col])),
                value=float(np.median(df[col])))
            sliders.append(slider)

    
    with col_2:
        
        st.markdown('Model Prediction')
        prob = rf.predict_proba(np.array(sliders)[None,:])
        prob_df = pd.DataFrame(data=prob, columns=['class_0','class_1','class_2'], index=['Probability'])
        st.write(prob_df)

        st.divider()

        col_1, col_2 = st.columns(2, gap="medium")

        with col_1:
            color_dict = {'class_0':'red','class_1':'violet','class_2':'green'}
            pred = rf.predict(np.array(sliders)[None,:])
            st.header(f"Predicts : :{color_dict[pred[0]]}[{pred[0]}]")
        with col_2:
            st.metric(
                label='Model Confidence', 
                value=f'{float(prob_df[pred[0]])*100 :0.2f}%', 
                delta=f'{(float(prob_df[pred[0]])-0.5)*100}%'
                )

        st.divider()
        st.subheader('Explanation')
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=x_train.to_numpy(), mode='classification', 
            class_names=['class_0','class_1','class_2'],
            feature_names=X.columns)
        explain = explainer.explain_instance(
            data_row=np.array(sliders), predict_fn=rf.predict_proba, 
            top_labels=3, num_features=X.shape[1])
        label_dict = {'class_0':0,'class_1':1,'class_2':2}
        fig = explain.as_pyplot_figure(label=label_dict[pred[0]])

        st.pyplot(fig)