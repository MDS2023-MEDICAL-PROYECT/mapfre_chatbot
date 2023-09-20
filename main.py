import streamlit as st
import streamlit_authenticator as stauth # pip install streamlit-authenticator
import yaml
from yaml.loader import SafeLoader
from src.classes.streamlit_app_chain import main

st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
    )
st.header("Your AI-Medical Assistant⚕️")

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp {margin-top: -50px}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


with open('./src/data/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

if "authenticator" not in st.session_state:
    st.session_state.authenticator = None
if "name" not in st.session_state:
    st.session_state.name = None
if "dni" not in st.session_state:
    st.session_state.dni = None


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'sidebar')


st.session_state.authenticator = authenticator
st.session_state.name = name
st.session_state.dni = username.upper()


if st.session_state["authentication_status"]:
    main()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


