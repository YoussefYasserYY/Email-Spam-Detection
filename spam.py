import streamlit as st
from emai_class import split,train,test,ex


st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


st.title('Email Spam Detection')

choice = st.sidebar.radio('',['Load Data','Split','Train','Test'],horizontal = True)

if choice == 'Load Data':
    try:
        from Data_profiler import profiler
        import pandas as pd
        df = pd.read_csv("spam.csv")
        st.session_state.df = df
        profiler(df,st)
    except:
        st.warning('No Data found')


if choice == 'Split':
    try:
        X,Y,x,y = split(st.session_state.df)
        st.session_state.X = X
        st.session_state.Y  =Y
        st.session_state.x = x
        st.session_state.y = y
    except:
        st.warning('Load Data First')
        
        
if choice == 'Train':
    try:
        st.header('Training')
        X= st.session_state.X
        Y  =st.session_state.Y
        x = st.session_state.x
        y = st.session_state.y
        df = st.session_state['df']
        train(X,Y,x,y)
    except:
        st.warning('Split Data First')
    
if choice == 'Test':
    try:
        test()
    except:
        st.warning('Fit Your Model First')
        
        
# if choice == 'Exercise':
#     st.title('Exercise')
#     ex('Email Spam Detection')