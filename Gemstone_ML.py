import streamlit as st
import joblib
import os 

import numpy as np 


attrib_info="""
#### Fields:
    - carat
    - cut
    - color
    - clarity
    - depth
    - table
    - x
    - y
    - z
"""

@st.cache(allow_output_mutation=True)

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def ml_app():
    st.subheader("Machine Learning Section")
    loaded_model=load_model(r'C:\Users\piyus\.spyder-py3\gemstonexgb.pkl')
    col1,col2=st.columns(2)
    with col1:
        carat = st.number_input("Carat",step=1e-2,format="%.2f")
        x = st.number_input("x",step=1e-2,format="%.2f")
        y = st.number_input("y",step=1e-2,format="%.2f")
        z = st.number_input("z",step=1e-2,format="%.2f")
        depth = st.number_input("depth",step=1e-2,format="%.2f")
        
    with col2:
        cut=st.selectbox('cut', [0,1,2,3,4])
        color=st.selectbox('color', [0,1,2,3,4,5,6])
        clarity=st.selectbox('clarity', [0,1,2,3,4,5,6,7])
        table = st.number_input("table",step=1e-2,format="%.2f")
        
    encoded_results=[carat,x,y,z,cut,color,clarity,depth,table]
    
    with st.expander('Predicted'):
        sample=np.array(encoded_results).reshape(1,-1)
        prediction = loaded_model.predict(sample)
        st.success('Predicted Price of the Gemstone: '+str(prediction[0]))

        
    
    