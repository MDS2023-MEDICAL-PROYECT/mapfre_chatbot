import langchain
import streamlit as st
import datetime
import database as db
import DoctorSummary_improved as doctor
from pathlib import Path
from PIL import Image
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from constants import OPENAI_API_KEY, INDEX_NAME, PINECONE_API_ENV, PINECONE_API_KEY
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
import pinecone
from langchain.chains.conversation.memory import ConversationBufferMemory


def clean_string(input):
    substring = ""

    if input is not None:
        start_pos = str(input).find(" additional_kwargs")
        if start_pos != -1:
            substring = str(input)[8:start_pos]

    return substring


def convert_to_string(messages):
    i = 0
    result = ""
    for message in messages:
        if i % 2 == 0:
            result += "-Human: " + clean_string(message) + "\n"
        else:
            result += "-Doctor: " + clean_string(message) + "\n"
        i = i + 1

    return result


def init_vectorstore():
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV,  # next to api key in console
    )
    index_name = INDEX_NAME

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=1536, metric="euclidean")

    index = pinecone.Index(index_name)
    return index


def get_vectordb():
    index = init_vectorstore()
    vectordb = Pinecone(
        index=index,
        embedding_function=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY).embed_query,
        text_key="text"
    )
    return vectordb


def get_conversation_chain(vectordb):
    llm = ChatOpenAI(temperature=0)
    retriever = vectordb.as_retriever()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    template = """You are an AI medical assistant. Based on our conversation and the relevant medical texts, 
    I'll suggest a follow-up question to help clarify or refine the diagnosis.
    
    When I formulate my question, I will:
    - Address you in the second person.
    - Avoid making references to specific places or figures from the medical texts.
    - Ensure not to repeat questions you've already answered.
    
    **Your Previous Answers**: {question} ----------- **Relevant Medical Texts**: {context} ----------- **My Next 
    Action**: If you've responded to five questions, I'll directly recommend a specialist for you. Otherwise, 
    I'll provide another question."""

    # añadir: No me hagas referencia a figuras, etc ... no me repitas preguntas si te las he contestado
    QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})

    return model


def main():

    langchain.debug = True
    # initial session_state in order to avoid refresh

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = get_vectordb()
    if "index" not in st.session_state:
        st.session_state.index = init_vectorstore()
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectordb)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "authenticator" not in st.session_state:
        st.session_state.authenticator = None
    if "name" not in st.session_state:
        st.session_state.name = None
    if "dni" not in st.session_state:
        st.session_state.dni = None

    # Sidebar for personal information
    with st.sidebar:
        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        profile_pic = current_dir / "assets" / "Bernardo.png"
        profile_pic = Image.open(profile_pic)
        patient = db.get_patient(st.session_state.dni)

        st.sidebar.header("Your Information")
        st.session_state.authenticator.logout('Logout', 'sidebar', key='unique_key')
        # st.sidebar.text(help="First Name", body=st.session_state.name)
        st.image(profile_pic, width=250)
        st.markdown(f"""
                **Nombre**: {st.session_state.name}\n
                **DNI**: {st.session_state.dni}\n  
                **Fecha de Nacimiento**: 15 de Agosto de 1983\n  
                **Sexo**: Masculino\n  
                **Dirección**: Calle Real No. 45, Ciudad Central\n  
                **Teléfono**: +52-555-1234567\n
                """)

        # st.session_state.authenticator.logout('Logout', 'sidebar', key='unique_key')
        if st.button(label="Coger Cita"):
            doctor.main()
        # last_name = st.sidebar.text_input("Last Name", "")
        # age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=0)

        # st.header("Symptoms")
        # symptoms = st.text_area("Check patient symptoms here", "")

        # st.header("Possible Diagnoses")
        # diagnosis_1 = st.text_input("Diagnosis 1", "")
        # diagnosis_2 = st.text_input("Diagnosis 2", "")
        # diagnosis_3 = st.text_input("Diagnosis 3", "")

    # if first_name and last_name and age > 0:
    personal_message = f"Hello {st.session_state.name}, how can I help you?"
    message(personal_message, is_user=False, avatar_style="big-smile")

    user_question = st.chat_input("Ask your question")

    if user_question:
        if not st.session_state.chat_history:
            pass
        else:
            for i, msg in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(msg.content, is_user=True, key=str(i) + '_user')
                else:
                    message(msg.content, is_user=False, key=str(i) + '_ai', avatar_style="big-smile")

        message(user_question, is_user=True, key=str(datetime.datetime.now()) + '_user')

        with st.spinner("Thinking..."):
            st.session_state.responses = st.session_state.conversation_chain({'question': user_question})
            #if "responses" not in st.session_state:
            #    st.session_state.responses = st.session_state.conversation_chain({'question': user_question})
            #else:
            #    conversation = convert_to_string(st.session_state.responses)
            #    user_question = conversation + "-Human: " + str(user_question) + "\n"
            #    st.session_state.responses = st.session_state.conversation_chain({'question': user_question})

            message(st.session_state.responses['answer'], is_user=False, key=str(datetime.datetime.now()) + '_ai',
                    avatar_style="big-smile")
            st.session_state.chat_history = st.session_state.responses['chat_history']

    # else:
    #    st.warning("Please enter valid personal information in the sidebar.")


if __name__ == '__main__':
    main()
else:
    print(__name__)
