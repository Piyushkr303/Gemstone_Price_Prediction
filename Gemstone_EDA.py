import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats


import matplotlib.pyplot as plt
import seaborn as sns

@st.cache(allow_output_mutation=True)

def retrive_data():
    df=pd.read_csv('cubic_zirconia.csv')
    df.drop('Unnamed: 0',axis=1)
    return df

def retrive_data2(df):
    numerical_data = df.select_dtypes(include='number')
    num_cols = numerical_data.columns
    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
    return df

def retrive_data3(df):
    df.drop('Unnamed: 0',axis=1)
    return df
    




def eda_app():
    df=retrive_data()
    st.subheader('Exploratory Data Analysis Section')
    submenu=st.sidebar.selectbox('Submenu',['Descriptive Statictics','Plots'])
    if submenu=='Descriptive Statictics':
        st.subheader('Descriptive Statictics')
        st.write("### Data Sample")
        st.dataframe(df.sample(50))
        
        with st.expander("Data Summary"):
            st.write("No of Rows: ",df.shape[0])
            st.write("No of Columns:",df.shape[1])
            
        with st.expander("Data Type Info"):
            st.dataframe(df.dtypes.T)
            
        with st.expander("Stats Columns"):
          #  st.dataframe(df.describe(include=np.object))
            st.dataframe(df.describe(include=np.number))
            
    if submenu=='Plots':
        st.subheader('Plots for EDA')
        
        with st.expander('Box Plot (Before Removal Of Outliers)'):
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(12, 5))
            fig.subplots_adjust(right=1.5)

            plt.subplot(1, 3, 1)
            sns.boxplot(y=df['carat'])

            plt.subplot(1, 3, 2)
            sns.boxplot(y=df['depth'])

            plt.subplot(1, 3, 3)
            sns.boxplot(y=df['table'])
            
            st.pyplot(fig)
            
        with st.expander("Box Plot After Romval Of Outliers"):
            df=retrive_data2(df)
            sns.set(style="whitegrid")  
            fig = plt.figure(figsize=(12, 5))
            fig.subplots_adjust(right=1.5)

            plt.subplot(1, 3, 1)
            sns.boxplot(y=df['x'])

            plt.subplot(1, 3, 2)
            sns.boxplot(y=df['y'])

            plt.subplot(1, 3, 3)
            sns.boxplot(y=df['z'])
            
            st.pyplot(fig)
            

        with st.expander("Analysis of Categorical Columns"):
            x = df['cut'].value_counts().values
            fig=plt.figure(figsize=(7, 6))
            plt.pie(x, center=(0, 0), radius=1.5, labels=df['cut'].unique(), autopct='%1.1f%%', pctdistance=0.5)
            plt.axis('equal')
            st.pyplot(fig)
            
            
        with st.expander("Bargraph of The Categorical Columns"):
            fig=plt.figure(figsize=(8, 5))
            sns.countplot(x='clarity', data=df)
            st.pyplot(fig)
            
            
        with st.expander("Average Prices Analysis"):
            df=retrive_data2(df)
            fig = plt.figure(figsize=(12, 5))
            fig.subplots_adjust(right=1.5)

            plt.subplot(1, 3, 1)
            df['price'].groupby(df['cut']).mean().plot(kind='barh', color='blue')

            plt.subplot(1, 3, 2)
            df['price'].groupby(df['color']).mean().plot(kind='barh', color='red')

            plt.subplot(1, 3, 3)
            df['price'].groupby(df['clarity']).mean().plot(kind='barh', color='green')
            
            st.pyplot(fig)
            
            
        with st.expander("Diagonostic Plot"):
           def diagnostic_plot(data, col):
               fig = plt.figure(figsize=(20, 5))
               fig.subplots_adjust(right=1.5)
   
               plt.subplot(1, 3, 1)
               sns.distplot(data[col], kde=True, color='teal')
               plt.title('Histogram')
   
               plt.subplot(1, 3, 2)
               stats.probplot(data[col], dist='norm', fit=True, plot=plt)
               plt.title('Q-Q Plot')
               plt.subplot(1, 3, 3)
               sns.boxplot(data[col],color='teal')
               plt.title('Box Plot')
   
               st.pyplot(fig)
   
           dist_lst=['carat', 'depth', 'table', 'x', 'y', 'z', 'price']

           for col in dist_lst:
               diagnostic_plot(df, col)
        
                            
        with st.expander("Correlation Plot"):
            df=retrive_data3(df)
            fig=plt.figure(figsize=(8, 5))
            df.corrwith(df['price']).plot(kind='barh', title="Correlation with 'Exited' column -") 
            st.pyplot(fig)
            
        with st.expander("Correlation Matrix"):
            df=retrive_data3(df)
            fig=plt.figure(figsize = (10, 8))
            corr = df.corr(method='spearman')
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
            cormat.set_title('Correlation Matrix')
            st.pyplot(fig)
            
        
