from typing import Final
import streamlit as st

# from os import environ
# from dotenv import load_dotenv
# load_dotenv()

PINECONE_API_KEY: Final[str] = st.secrets["pinecone_api_key"]
PINECONE_API_ENV: Final[str] = st.secrets["pinecone_api_env"]
OPENAI_API_KEY: Final[str] = st.secrets["openai_api_key"]
DETA_KEY: Final[str] = st.secrets["deta_key"]
INDEX_NAME: Final[str] = 'thevalleyopenai'