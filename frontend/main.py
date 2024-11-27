import requests
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import requests
import json
import os

def map_drought_category(prediction):
    categories = {
        0: ("D0", "Abnormally Dry", "Short-term dryness slowing planting, growth of crops or pastures."),
        1: ("D1", "Moderate Drought", "Some damage to crops, streams, and voluntary water-use restrictions."),
        2: ("D2", "Severe Drought", "Crop or pasture losses likely, water shortages common."),
        3: ("D3", "Extreme Drought", "Major crop/pasture losses, widespread water shortages."),
        4: ("D4", "Exceptional Drought", "Exceptional and widespread crop/pasture losses and water emergencies."),
    }
    return categories.get(prediction, ("Unknown", "Unknown", "No description available"))

# Activer le mode large
st.set_page_config(layout="wide")


# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Home","Predict", "Predict/CSV"])

# Navigation entre les pages
if page == "Home":
    # defines an h1 header
    st.title("Welcome Drought prediction App🌵")
    st.write(
        """
        ## Welcome to the Drought Prediction App using Machine Learning.
        
        ### Features:
        - **Predict from User Input**: Provide climate data to get a prediction.
        - **Predict from a CSV File**: Upload a CSV file with your data and get results.

        This app leverages machine learning algorithms to analyze climate data and predict whether a drought is likely.
        """
    )
    st.image(
        "https://news.stanford.edu/__data/assets/image/0025/36808/Climate-change-from-drought-to-green-growth.jpeg",
        caption="Analyse des conditions climatiques pour la prédiction de la sécheresse.",
        use_container_width=True,
    )

elif page == "Predict":
    st.title("Drought prediction from user input")
    st.subheader("Enter details below: ")

    # Formulaire avec deux colonnes
    with st.form("form", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns([1,1,1,1])

        # Champs dans la première colonne
        with col1:
            fips = st.number_input("Enter FIPS code", value=1001, step=1)
            date = st.date_input("Enter date", datetime.date(2000, 1, 4))
            PRECTOT = st.number_input("Enter PRECTOT value", value=15.95)
            PS = st.number_input("Enter PS value", value=100.29)
            QV2M = st.number_input("Enter QV2M value", value=6.42)
                    

        # Champs dans la deuxième colonne
        with col2:
            T2M = st.number_input("Enter T2M value", value=11.4)
            T2MDEW = st.number_input("Enter T2MDEW value", value=6.09)
            T2MWET = st.number_input("Enter T2MWET value", value=6.1)
            T2M_MAX = st.number_input("Enter T2M_MAX value", value=18.09)
            T2M_MIN = st.number_input("Enter T2M_MIN value", value=2.16)
        
        with col3:
            T2M_RANGE = st.number_input("Enter T2M_RANGE value", value=15.92)
            TS = st.number_input("Enter TS value", value=11.31)
            WS10M = st.number_input("Enter WS10M value", value=3.84)
            WS10M_MAX = st.number_input("Enter WS10M_MAX value", value=5.67)
            WS10M_MIN = st.number_input("Enter WS10M_MIN value", value=2.08)
        with col4:
            WS10M_RANGE = st.number_input("Enter WS10M_RANGE value", value=3.59)
            WS50M = st.number_input("Enter WS50M value", value=6.73)
            WS50M_MAX = st.number_input("Enter WS50M_MAX value", value=9.31)
            WS50M_MIN = st.number_input("Enter WS50M_MIN value", value=3.74)
            WS50M_RANGE = st.number_input("Enter WS50M_RANGE value", value=5.58)


        # Création du dictionnaire
        dd = {
            "fips": fips,
            "date": date.strftime("%Y-%m-%d"),
            "PRECTOT": PRECTOT,
            "PS": PS,
            "QV2M": QV2M,
            "T2M": T2M,
            "T2MDEW": T2MDEW,
            "T2MWET": T2MWET,
            "T2M_MAX": T2M_MAX,
            "T2M_MIN": T2M_MIN,
            "T2M_RANGE": T2M_RANGE,
            "TS": TS,
            "WS10M": WS10M,
            "WS10M_MAX": WS10M_MAX,
            "WS10M_MIN": WS10M_MIN,
            "WS10M_RANGE": WS10M_RANGE,
            "WS50M": WS50M,
            "WS50M_MAX": WS50M_MAX,
            "WS50M_MIN": WS50M_MIN,
            "WS50M_RANGE": WS50M_RANGE,
        }

        

        # Bouton de soumission
        submit = st.form_submit_button("Submit this form")
        if submit:
            res = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(dd))
            predictions = res.json().get("predictions")
            st.write(map_drought_category(predictions[0]))

else:
    st.title("Drought prediction from csv file")
    st.subheader("Enter your drought csv file")
    data = st.file_uploader("Choose a csv file")


    # displays a button
    if data is not None:
        file = {"file": data.getvalue()}
        res = requests.post("http://127.0.0.1:8000/predict/csv", files=file)
        predictions = res.json().get("predictions")
        st.text(predictions)