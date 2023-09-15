from typing import Final
import streamlit as st

PINECONE_API_KEY: Final[str] = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV: Final[str] = st.secrets["PINECONE_API_ENV"]
OPENAI_API_KEY: Final[str] = st.secrets["OPENAI_API_KEY"]
DETA_KEY: Final[str] = st.secrets["DETA_KEY"]
INDEX_NAME: Final[str] = 'thevalleyopenai'
ITERATIONS = 4

