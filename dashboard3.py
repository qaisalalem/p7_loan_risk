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
    cat_features = list(data.select_dtypes('object').nunique().index)

    # Encoding categorical features
    df = pd.get_dummies(X_df, columns=cat_features)

    # Numerical and binary features
    features_df = df.nunique()
    num_features = list(features_df[features_df != 2].index)
    binary_features = list(features_df[features_df == 2].index)
    df['NAME_FAMILY_STATUS_Unknown'] = 0
    binary_features.append('NAME_FAMILY_STATUS_Unknown')

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



def customer_description(data):
    """This function creates a dataframe with customer descriptions."""
    df = pd.DataFrame(
        columns=['Gender', 'Age (years)', 'Family status',
                 'Number of children', 'Days employed',
                 'Income ($)', 'Credit amount ($)', 'Loan annuity ($)'])
    data['AGE'] = data['DAYS_BIRTH'] / 365
    df['Customer ID'] = list(data.SK_ID_CURR.astype(str))
    df['Gender'] = list(data.CODE_GENDER)
    df['Age (years)'] = list(data.AGE.abs().astype('int64'))
    df['Family status'] = list(data.NAME_FAMILY_STATUS)
    df['Number of children'] = list(data.CNT_CHILDREN.astype('int64'))
    df['Days employed'] = list(data.DAYS_EMPLOYED.abs().astype('int64'))
    df['Income ($)'] = list(data.AMT_INCOME_TOTAL.astype('int64'))
    df['Credit amount ($)'] = list(data.AMT_CREDIT.astype('int64'))
    df['Loan annuity ($)'] = list(data.AMT_ANNUITY.astype('int64'))
    df['Organization type'] = list(data.ORGANIZATION_TYPE)
    return df


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
    customer_df = data[data.SK_ID_CURR == customer_id]
    viz_df = customer_df.round(2)
    st.write(viz_df)

    # Preprocessed customer data for prediction
    X = norm_df[norm_df.SK_ID_CURR == customer_id]
    X = X.drop(['SK_ID_CURR'], axis=1)
    
    
    #Visualisation according to new advice
    #dropdown menu for to graphs, correlation between selected variables
    variables_list1= list(data.columns)
    variable1= st.sidebar.selectbox(
        "Please select variable #1 :", variables_list1)


    variables_list2= list(data.columns)
    variable2= st.sidebar.selectbox(
        "Please select variable #2 :", variables_list2)

    #st.subheader('Graph showing total income of all clients in the database')
    #i am using amt_inc_total to show selected client.
    amt_inc_total = np.log(data.loc[data['SK_ID_CURR'] == int(customer_id), 'AMT_INCOME_TOTAL'].values[0])
    #x_a = [np.log(data['AMT_INCOME_TOTAL'])]
    #fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
    #fig_a.add_vline(x=amt_inc_total, annotation_text=' Selected client')
    #st.plotly_chart(fig_a, use_container_width=True)

    #visualisation fig 1
    #st.write(variable1)
    st.subheader('Graph showing variable 1')
    df = data[variable1] #i managed to select my variable now i need to plot it. this method works, i need to try another method
    df=[np.log(df)]
    
    fig=ff.create_distplot(df, [variable1] , bin_size= 0.3)
    fig.add_vline(x=amt_inc_total, annotation_text=' Selected client')
    st.plotly_chart(fig, use_container_width=True)

    
    

    #Visualisation fig 2
    df2=data[variable2]
    df2=[np.log(df2)]
    #df2=pd.Series(df2)
    st.subheader('Graph showing variable 2')
    fig_b=ff.create_distplot(df2, [variable2] , bin_size= 0.3)
    fig_b.add_vline(x=amt_inc_total, annotation_text=' Selected client')
    st.plotly_chart(fig_b, use_container_width=True)
    
    #dataframe=norm_df.copy(deep=True)
    #dataframe=dataframe.select_dtypes(include=['float64', 'int64'], exclude='bool')
    #dataframe=dataframe.replace([np.inf, -np.inf], np.nan)

    #st.write(dataframe)
    #trying to display processed data as figures.
 #   variables_list3= list(dataframe.columns)
 #   variable3= st.sidebar.selectbox(
  #      "Please select variable #3 :", variables_list3)

#####TypeError: Cannot compare types 'ndarray(dtype=object)' and 'float'
 
 #   df3=dataframe[variable3]
    #df3=[df3]
 #   df3=[np.log(df3)]
 #   df3=pd.Series(df3)
    #df3=df3.transform(np.log)
    #st.write(dataframe)
  #  df3=df3.dropna(inplace=True)
    #df3.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df3.dropna(inplace=True)
    #df3=df3[df3.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 

    #df3=[v for v in df3 if not m.isnan(v) and not m.isinf(v)] 
    
    st.subheader('Graph showing variable 3')
    #fig_c=ff.create_distplot(df3, [variable3] , bin_size= 0.3)
    #fig_c.add_vline(x=amt_inc_total, annotation_text=' Selected client')
    #st.plotly_chart(fig_c, use_container_width=True)



    st.header('''Credit application result''')


    #prediction
    y_proba = model.predict_proba(np.array(X))[0][1]

    # Looking for the customer situation (class 0 or 1)
    # by using the best threshold from precision-recall curve
    y_class = round(y_proba, 2)
    best_threshold = 0.36
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
             "Clients with a score greater than *36* are at risk of default.**")
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
                                value = y_proba,
                                number = {'font':{'size':48}},
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Customer's Request Status", 'font': {'size': 28, 'color':color(customer_class)}},
                                delta = {'reference': best_threshold, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
                                gauge = {'axis': {'range': [0,1], 'tickcolor': color(customer_class)},
                                         'bar': {'color': color(customer_class)},
                                         'steps': [{'range': [0,best_threshold], 'color': 'lightgreen'},
                                                    {'range': [best_threshold,1], 'color': 'lightcoral'}],
                                         'threshold': {'line': {'color': "black", 'width': 5},
                                                       'thickness': 1,
                                                       'value': best_threshold}}))
    st.plotly_chart(fig)

    
    
    if status=='accepted':
        original_title = '<p style="font-family:Courier; color:GREEN; font-size:65px; text-align: center;">{}</p>'.format(customer_class,": Client's loan application is successful")
        st.markdown(original_title, unsafe_allow_html=True)
    else :
        original_title = '<p style="font-family:Courier; color:red; font-size:65px; text-align: center;">{}</p>'.format(customer_class,": Client's loan application is unsuccessful")
        st.markdown(original_title, unsafe_allow_html=True)
    
    
    
    

    # Feature importance
    model.predict(np.array(X_norm))
    features_importance = model.feature_importances_
    sorted = np.argsort(features_importance)
    dataviz = pd.DataFrame(columns=['feature', 'importance'])
    dataviz['feature'] = np.array(X_norm.columns)[sorted]
    dataviz['importance'] = features_importance[sorted]
    dataviz = dataviz[dataviz['importance'] > 50]
    dataviz.reset_index(inplace=True, drop=True)
    dataviz = dataviz.sort_values(['importance'], ascending=False)

    # SHAP explanations
    shap.initjs()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X)
    shap_df = pd.DataFrame(
        list(zip(X.columns, np.abs(shap_values[0]).mean(0))),
        columns=['feature', 'importance'])
    shap_df = shap_df.sort_values(by=['importance'], ascending=False)
    shap_df.reset_index(inplace=True, drop=True)
    shap_features = list(shap_df.iloc[0:20, ].feature)

    #plotting global feature importance.
    st.header("Interprétabilité globale du modèle")
    fig9 = plt.figure(figsize=(10, 20))
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
                        max_display=50,
                        plot_type='bar',
                        plot_size=(5, 15))
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
