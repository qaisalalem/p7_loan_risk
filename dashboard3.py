# Project 7: Implementing a scoring model
# Import libraries
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time
import lightgbm as lgb
import plotly.graph_objects as go

# Serialization library
import pickle

# Front end library
import streamlit as st

# Visualization library
import plotly_express as px

# SHAP library
import shap

from pathlib import Path
import plotly.figure_factory as ff

plt.style.use('seaborn')
import math as m


def load_data(file):
    """This function is used to load the dataset."""
    folder = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(folder, file)
    data = pd.read_csv(data_path, encoding_errors='ignore')
    return data


def preprocessing(data, num_imputer, bin_imputer, transformer, scaler):
    """This function is used to perform data preprocessing."""
    X_df = data.drop(['SK_ID_CURR'], axis=1)

    # Feature selection
    # Categorical features
    
    numeric_features = list(X_df.select_dtypes('int64').nunique().index)
    numeric_features.extend(list(X_df.select_dtypes('float64').nunique().index))
    cat_features = list(X_df.select_dtypes('object').nunique().index)

    # Encoding categorical features
    df = pd.get_dummies(X_df, columns=cat_features)

    # Numerical and binary features
    features_df = df.nunique()
    num_features = list(features_df[features_df != 2].index)
    binary_features = list(features_df[features_df == 2].index)
    #df['NAME_FAMILY_STATUS_Unknown'] = 0
    #binary_features.append('NAME_FAMILY_STATUS_Unknown')
    #st.write(len(binary_features))
    #st.write(binary_features)

    # Imputations
    X_num = pd.DataFrame(num_imputer.transform(df[num_features]),
                         columns=num_features)
    X_bin = pd.DataFrame(bin_imputer.transform(df[binary_features]),
                         columns=binary_features)

    # Normalization
    X_norm = pd.DataFrame(transformer.transform(X_num), columns=num_features)

    # Standardization
    norm_df = pd.DataFrame(scaler.transform(X_norm), columns=num_features)

    for feature in binary_features:
        norm_df[feature] = X_bin[feature]
    norm_df['SK_ID_CURR'] = data['SK_ID_CURR']
    return norm_df



def load_model(file, key):
    """This function is used to load a serialized file."""
    path = open(file, 'rb')
    model_pickle = pickle.load(path)
    model = model_pickle[key]
    return model






def main():
    st.set_page_config(layout='wide')
    st.title("CREDIT SCORING APPLICATION")
    st.subheader('Data of selected client')
    
    # Loading the dataset
    data = load_data('data/data.csv')

    # Loading the model
    model = load_model('model/model.pkl', 'model')

    # Loading the numerical imputer
    num_imputer = load_model('model/num_imputer.pkl', 'num_imputer')

    # Loading the binary imputer
    bin_imputer = load_model('model/bin_imputer.pkl', 'bin_imputer')

    # Loading the numerical transformer
    transformer = load_model('model/transformer.pkl', 'transformer')

    # Loading the numerical scaler
    scaler = load_model('model/scaler.pkl', 'scaler')

    # Preprocessing
    norm_df = preprocessing(data,
                            num_imputer,
                            bin_imputer,
                            transformer,
                            scaler)
    X_norm = norm_df.drop(['SK_ID_CURR'], axis=1)

    # Customer selection
    customers_list = list(data.SK_ID_CURR)
    customer_id = st.sidebar.selectbox(
        "Please select client ID :", customers_list)
    
    # Customer data
    relevant_features= ['POS_SK_DPD_DEF','BUR_DAYS_CREDIT_ENDDATE','BUR_AMT_CREDIT_SUM','BUR_AMT_CREDIT_SUM_DEBT',
                        'BUR_AMT_CREDIT_SUM_OVERDUE','BUR_DAYS_CREDIT_UPDATE','PAY_HIST_NUM_INSTALMENT_VERSION',
                        'PAY_HIST_NUM_INSTALMENT_NUMBER','PAY_HIST_DAYS_INSTALMENT','PAY_HIST_AMT_INSTALMENT',
                        'POS_CNT_INSTALMENT','POS_SK_DPD','NAME_EDUCATION_TYPE_Secondary / secondary special',
                        'PREV_APPLICATION_NUMBER','PREV_AMT_ANNUITY','PREV_AMT_DOWN_PAYMENT',
                        'PREV_AMT_CREDIT','PREV_RATE_DOWN_PAYMENT','PREV_CNT_PAYMENT',
                        'NAME_CONTRACT_TYPE','FLAG_OWN_CAR','NAME_EDUCATION_TYPE_Higher education',
                        'REG_CITY_NOT_LIVE_CITY','CODE_GENDER_F','NAME_FAMILY_STATUS_Married',
                        'BUR_DAYS_CREDIT','BUR_CNT_CREDIT_PROLONG','REGION_RATING_CLIENT',
                        'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',
                        'PAYMENT_RATE','HOUR_APPR_PROCESS_START','EXT_SOURCE_2',
                        'EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
                        'DAYS_LAST_PHONE_CHANGE','AMT_ANNUITY','AMT_CREDIT',
                        'AMT_INCOME_TOTAL','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',
                        'ANNUITY_INCOME_RATE','INCOME_CREDIT_RATE','DAYS_BIRTH',
                        'REGION_POPULATION_RELATIVE']
    customer_df = data[data.SK_ID_CURR == customer_id]
    viz_df = customer_df.round(2)
    st.write(viz_df)

    # Preprocessed customer data for prediction
    X = norm_df[norm_df.SK_ID_CURR == customer_id]
    
    X = X.drop(['SK_ID_CURR'], axis=1)
    X= X[relevant_features]
    
    
    #Visualisation according to new advice
    dash=data.drop(['SK_ID_CURR'], axis=1)
    importantvar=['PAYMENT_RATE','DAYS_BIRTH','EXT_SOURCE_2',
                  'EXT_SOURCE_3','PAY_HIST_DAYS_INSTALMENT','AMT_INCOME_TOTAL',
                  'AMT_ANNUITY','AMT_CREDIT','PREV_AMT_ANNUITY','ANNUITY_INCOME_RATE',
                  'DAYS_EMPLOYED'
                 ]
    dash=dash[importantvar]
    dash['AMT_INCOME_TOTAL']=np.log(dash['AMT_INCOME_TOTAL']+1)
    dash['ANNUITY_INCOME_RATE']=np.log(dash['ANNUITY_INCOME_RATE']+1)
    dash['PREV_AMT_ANNUITY']=np.log(dash['PREV_AMT_ANNUITY']+1)
    dash['AMT_ANNUITY']=np.log(dash['AMT_ANNUITY']+1)
    dash['AMT_CREDIT']=np.log(dash['AMT_CREDIT']+1)
    dash['DAYS_EMPLOYED']=np.sqrt(dash['DAYS_EMPLOYED']*-1)
    dash['AGE'] = dash['DAYS_BIRTH'] / 365
    dash['AGE']= dash.AGE.abs().astype('int64')
    
    
    
    
    
    st.header('''Credit application result''')
    #prediction
    y_proba = model.predict_proba(np.array(X))[0][1]

    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.37
    customer_class = np.where(y_class > best_threshold, 1, 0)

    # Customer score calculation
    score = int(y_class * 100)

    # Customer credit application result
    if customer_class == 1:
        result = 'at risk of default'
        status = 'refused'
    else:
        result = 'no risk of default'
        status = 'accepted'
    
    #prediction
    st.write("* **The credit score is between 0 & 100. "
             "Clients with a score greater than *37* are at risk of default.**")
    st.write("* **Class 0: client does not default**")
    st.write("* **Class 1: client defaults**")
    st.write("Client N°{} credit score is **{}**. "
                 "The client is classified as **{}**, "
                 "the credit application is **{}**.".format(customer_id, score,
                                                       customer_class, status))
    if customer_class == 0: 
        st.success("Client's loan application is successful :thumbsup:")
    else: 
        st.error("Client's loan application is unsuccessful :thumbsdown:") 

    #visualisation showing score and threshold
    def color(status):
        '''Définition de la couleur selon la prédiction'''
        if status=='accepted':
            col='Green'
        else :
            col='Red'
        return col
    fig = go.Figure(go.Indicator(mode = "gauge+number+delta",
                                value = score,
                                number = {'font':{'size':48}},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Customer's Request Status", 'font': {'size': 28, 'color':color(customer_class)}},
                                delta = {'reference': (best_threshold *100), 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
                                gauge = {'axis': {'range': [0,100], 'tickcolor': color(customer_class)},
                                         'bar': {'color': color(customer_class)},
                                         'steps': [{'range': [0,(best_threshold *100)], 'color': 'lightgreen'},
                                                    {'range': [(best_threshold *100),100], 'color': 'lightcoral'}],
                                         'threshold': {'line': {'color': "black", 'width': 5},
                                                       'thickness': 1,
                                                       'value': (best_threshold *100)}}))
    st.plotly_chart(fig)

    
    
    if status=='accepted':
        original_title = '<p style="font-family:Courier; color:GREEN; font-size:65px; text-align: center;">Loan is accepted</p>'#.format()
        st.markdown(original_title, unsafe_allow_html=True)
    else :
        original_title = '<p style="font-family:Courier; color:red; font-size:65px; text-align: center;">Loan is refused</p>'#.format()
        st.markdown(original_title, unsafe_allow_html=True)
    
    #dropdown menu for the graphs
    variables_list1= list(dash.columns)
    variable1= st.sidebar.selectbox(
        "Please select variable #1 :", variables_list1)


    variables_list2= list(dash.columns)
    variable2= st.sidebar.selectbox(
        "Please select variable #2 :", variables_list2)

    
    amt_inc_total = (dash.loc[data['SK_ID_CURR'] == int(customer_id), variable1].values[0])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    
    #visualisation fig 1
    st.subheader('Graph showing variable 1')
    a=sns.distplot(dash[variable1], bins=30, (figsize=(15,8))
    a.axvline(x=amt_inc_total)
    st.pyplot()

    

    #Visualisation fig 2
    amt_inc_total2 = (dash.loc[data['SK_ID_CURR'] == int(customer_id), variable2].values[0])
    st.subheader('Graph showing variable 2')
    b=sns.distplot(dash[variable2], bins=30)
    b.axvline(x=amt_inc_total2)
    st.pyplot()
   
    #Visualisation fig 3
    st.subheader('Graph showing scatterplot between the 2 selected variables')
    c=sns.scatterplot(x=dash[variable1], y=dash[variable2])
    st.pyplot()




  
    # Feature importance
    model.predict(np.array(X_norm[relevant_features]))
    features_importance = model.feature_importances_
    sorted = np.argsort(features_importance)
    dataviz = pd.DataFrame(columns=['feature', 'importance'])
    dataviz['feature'] = np.array(X_norm[relevant_features].columns)[sorted]
    dataviz['importance'] = features_importance[sorted]
    dataviz = dataviz[dataviz['importance'] > 200]
    dataviz.reset_index(inplace=True, drop=True)
    dataviz = dataviz.sort_values(['importance'], ascending=False)

    # SHAP explanations
    shap.initjs()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X)
    shap_df = pd.DataFrame(
        list(zip(X[relevant_features].columns, np.abs(shap_values[0]).mean(0))),
        columns=['feature', 'importance'])
    shap_df = shap_df.sort_values(by=['importance'], ascending=False)
    shap_df.reset_index(inplace=True, drop=True)
    shap_features = list(shap_df.iloc[0:20, ].feature)

    #plotting global feature importance.
    st.header("Interprétabilité globale du modèle")
    fig9 = plt.figure(figsize=(10, 10))
    sns.barplot(x='importance', y='feature', data=dataviz)
    st.write("Le RGPD (article 22) prévoit des règles restrictives"
                 " pour éviter que l’homme ne subisse des décisions"
                 " émanant uniquement de machines.")
    st.write("L'interprétabilité globale permet de connaître de manière"
                 " générale les variables importantes pour le modèle. ")
    st.write("L’importance des variables ne varie pas"
                 " en fonction des données de chaque client.")
    st.write(fig9)

    #plotting local feature importance.
    st.header("Interprétabilité locale du modèle")
    fig10 = plt.figure()
    shap.summary_plot(shap_values, X,
                        feature_names=list(X.columns),
                        max_display=15,
                        plot_type='bar',
                        plot_size=(5, 5))
    st.write("Le RGPD (article 22) prévoit des règles restrictives"
                 " pour éviter que l’homme ne subisse des décisions"
                 " émanant uniquement de machines.")
    st.write("SHAP répond aux exigences du RGPD et permet de déterminer"
                 " les effets des différentes variables dans le résultat de la"
                 " prédiction du score du client N°{}.".format(customer_id))
    st.write("L’importance des variables varie en fonction"
                 "  des données de chaque client.")
    st.pyplot(fig10)


    
if __name__ == '__main__':
    main()
