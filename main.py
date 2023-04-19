import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# loaded_model = pickle.load(open('pretrained_model.sav', 'rb'))
#kmean = pickle.load(open('kmeans.pkl', 'rb'))
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = model['scaler']
pca = model['pca']
kmean = model['kmean']

st.title('World Development Clusters')
st.sidebar.header('User Input Features')

def cluster_prediction(data):
    input = data
    input_data_as_numpy_array = np.asarray(input)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    X_new_scaled = scaler.transform(input_data_reshaped)
    X_new_pca = pca.transform(X_new_scaled)
    prediction = kmean.predict(X_new_pca)
    if prediction[0] == 0:
        return 'Developing Country'
    elif prediction[0] == 1:
        return 'Developed Country'
    elif prediction[0] == 2:
        return 'Underdeveloped Country'
    elif prediction[0] == 3:
        return 'Moderately Developed Country'


def get_input():
    Birth_Rate = st.sidebar.slider('Birth Rate', 0.007, 0.053, 0.001)
    CO2_Emissions = st.sidebar.number_input('C02 Emissions')
    GDP = st.sidebar.number_input('Total GDP')
    Health_Exp_GDP = st.sidebar.slider('Health Exp % GDP', 0.02, 0.9, 0.45)
    Health_Exp_Capita = st.sidebar.slider('Health Exp/Capita', 2, 9988, 100)
    Infant_Mortality_Rate = st.sidebar.slider('Infant Mortality Rate', 0.002, 0.141, 0.05)
    Internet_Usage = st.sidebar.slider('Internet Usage', 0.0, 1.0, 0.1)
    Life_Expectancy_Female = st.sidebar.slider('Life Expectancy Female', 1, 99, 70)
    Life_Expectancy_male = st.sidebar.slider('Life Expectancy Male', 1, 99, 70)
    Mobile_Phone_Usage = st.sidebar.slider('Mobile Phone Usage', 0.0, 2.9, 1.0)
    Population_0_14 = st.sidebar.slider('Population 0-14 %', 0.1, 0.5, 0.2)
    Population_15_64 = st.sidebar.slider('Population 15-64 %', 0.2, 0.8, 0.4)
    Population_65above = st.sidebar.slider('Population 65% +', 0.001, 0.4, 0.05)
    Population_Total = st.sidebar.number_input('Total population')
    Population_Urban = st.sidebar.slider('Population Urban %', 0.082, 1.0, 0.5)
    Tourism_Inbound = st.sidebar.number_input('Tourism inbound')
    Tourism_outbound = st.sidebar.number_input('Tourism outbound')
    Business_Tax_Rate = st.sidebar.number_input('Business Tax Rate')
    Days_to_Start_Business = st.sidebar.slider('Days to Start Business', 1, 694, 10)
    Ease_of_Business = st.sidebar.number_input('Ease of Business')
    Energy_Usage = st.sidebar.number_input('Energy Usage')
    Hours_todo_Tax = st.sidebar.number_input('Hours to do Tax')
    Lending_Interest = st.sidebar.number_input('Lending Interest')


    data = {'Birth_Rate': Birth_Rate, 'CO2_Emissions': CO2_Emissions, 'GDP': GDP,
             'Health_Exp_GDP': Health_Exp_GDP,'Health_Exp_Capita': Health_Exp_Capita,
            'Infant_Mortality_Rate': Infant_Mortality_Rate, 'Internet_Usage': Internet_Usage,
            'Life_Expectancy_Female': Life_Expectancy_Female, 'Life_Expectancy_male': Life_Expectancy_male,
            'Mobile_Phone_Usage ': Mobile_Phone_Usage, 'Population_0_14': Population_0_14,
            'Population_15_64': Population_15_64,
            'Population_65above': Population_65above, 'Population_Total': Population_Total,
            'Population_Urban': Population_Urban,
            'Tourism_Inbound': Tourism_Inbound, 'Tourism_outbound': Tourism_outbound,
            'Business_Tax_Rate': Business_Tax_Rate ,
            'Days_to_Start_Business': Days_to_Start_Business, 'Ease_of_Business': Ease_of_Business,
            'Energy_Usage': Energy_Usage,
            'Hours_todo_Tax': Hours_todo_Tax, 'Lending_Interest': Lending_Interest}

    features = pd.DataFrame(data, index=[0])
    prediction = cluster_prediction(features)
    return features,prediction

data, prediction = get_input()

if st.button("Submit"):
    st.write("Input Features:")
    st.write(data)
    st.write("Prediction:")
    st.success(prediction)







