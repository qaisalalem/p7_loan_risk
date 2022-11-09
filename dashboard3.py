# Project 7: Implementing a scoring model
# Import libraries
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests
import time

# Serialization library
import pickle

# Front end library
import streamlit as st

# Visualization library
import plotly_express as px


from pathlib import Path
import plotly.figure_factory as ff

plt.style.use('seaborn')


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
    
    

    #visualisation
    st.subheader('Graph showing total income of all clients in the database')
    amt_inc_total = np.log(data.loc[data['SK_ID_CURR'] == int(customer_id), 'AMT_INCOME_TOTAL'].values[0])
    x_a = [np.log(data['AMT_INCOME_TOTAL'])]
    fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
    fig_a.add_vline(x=amt_inc_total, annotation_text=' Selected client')

    st.plotly_chart(fig_a, use_container_width=True)

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
    st.write("**The credit score is between 0 & 100. "
             "Clients with a score greater than *36* are at risk of default.**")
    st.write("**Class 1: client defaults**")
    st.write("Client N°{} credit score is **{}**. "
                 "The client is classified as **{}**, "
                 "the credit application is **{}**.".format(customer_id, score,
                                                       customer_class, status))
    if customer_class == 0: 
        st.success("Client's loan application is successful :thumbsup:")
    else: 
        st.error("Client's loan application is unsuccessful :thumbsdown:") 


if __name__ == '__main__':
    main()
