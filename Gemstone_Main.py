import streamlit as st
import streamlit.components.v1 as stt

from Gemstone_EDA import eda_app
from Gemstone_ML import ml_app

def main():
    st.title('Gemstone Price Exploratory Data Analysis')
    st.header('Supervised Machine Learning Model')
    menu=['Home','EDA','Machine Learning','About']
    choice=st.sidebar.selectbox('Menu',menu)
    if choice=='Home':
        st.subheader('Gemstone Dataset')
        st.write('')
        st.write("""The Task was to perform EDA upon the given dataset and find the underlying 
                 datatrends along with the pridictive modeling to 
                 make predications of the new data when given to the model.
                 
                 The steps involved in the Modeling of the data
                 
                 1. EDA and Analysis
                 
                 2. Feature Enginnering
                 
                 3. Data Scaling
                 
                 4. Predictive Modeling

                 
                 Our Model uses the Ensemble Technique XGBoost 
                 Pregressor for making the prediction with a 
                 Train Test Accuracy of :--
                 
                 Train Accuracy :- 99.0 %
                 Test Accuracy :- 98.0 %
                 
                 """)
        st.write('')
        
    elif choice=='EDA':
        eda_app()
        
    elif choice=='Machine Learning':
        ml_app()
        
    else:
        st.subheader('About')
        st.subheader('Created by -PIYUSH KUMAR')
        st.markdown('Branch :- Information Technology')
        st.markdown('This Project is created by Piyush Kumar, Student of Kamla Nehru Institute of Technology, Sultanpur')
        
main()

# streamlit run C:\Users\piyus\.spyder-py3\Gemstone_Main.py