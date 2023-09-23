import json
import langchain
import pinecone
import streamlit as st
import datetime
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from src.clients.database import DetaClient
from pathlib import Path
from PIL import Image
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from src.constants.medic_bot import MedicBotConstants
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory


def translate_english_spanish(phrase):
    llm = ChatOpenAI(temperature=0, openai_api_key=MedicBotConstants.OPENAI_API_KEY)
    prompt_template = PromptTemplate.from_template(
        """Your are a translator from english to Spanish or from Spanish to English. Translate the following phrase. 
        Your answer must be direct.

        {chat}
        """

    )
    prompt_template.format(chat=phrase)
    translation_model = LLMChain(llm=llm, prompt=prompt_template, output_key="translation")

    return translation_model(phrase)


def init_vectorstore():
    pinecone.init(
        api_key=MedicBotConstants.PINECONE_API_KEY,  # find at app.pinecone.io
        environment=MedicBotConstants.PINECONE_API_ENV,  # next to api key in console
    )
    index_name = MedicBotConstants.INDEX_NAME

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(MedicBotConstants.INDEX_NAME, dimension=1536, metric="euclidean")

    index = pinecone.Index(index_name)
    return index


def get_vectordb():
    index = init_vectorstore()
    vectordb = PineconeHybridSearchRetriever(
        embeddings=OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=MedicBotConstants.OPENAI_API_KEY)
        , sparse_encoder=BM25Encoder().default(), index=index, top_k=5, alpha=0.5
    )
    return vectordb


def get_conversation_chain(vectordb):
    llm = ChatOpenAI(temperature=0)
    retriever = vectordb

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    template = """As a virtual medical assistant, your role is to facilitate a detailed understanding of the 
    patient's current health condition. Refer to the conversation history to identify the symptoms explicitly 
    mentioned by the patient so far. 
    Based on the medical texts and the patient's symptoms mentioned, 
    the doctor should inquire about any additional symptoms that may be related to the diagnoses found in the medical 
    texts. Please ask the patient a question to determine if they are experiencing any symptom that has not been 
    mentioned in the chat history but is associated with the diagnoses from the medical texts. Be sure to phrase your 
    question in a way that fosters a direct and personalized engagement with the patient.

    Please adhere to the following guidelines: 
    - Pose only one follow-up question in each interaction to maintain a focused and fruitful conversation. 
    - Ask only about one specific symptom 
    - Formulate your question from a second person perspective to foster a direct and personalized engagement with the 
    patient. 
    - Avoid making explicit references to the medical texts in your questions to maintain a natural conversation flow. 
    - Refrain from referring to specific details from the medical texts to avoid confusion. 
    - Avoid asking questions that have already been addressed in the conversation. 
    - Be kind and maintain a natural conversation flow
    
    **Your Previous Responses**: {question} ----------- **Relevant Medical Texts**: {context} -----------
    
    Your task is to ask as you were the doctor, without including reasoning. Please provide a follow-up question
    
    """

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
    standalone question, in its original language.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""

    QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})

    return model


def get_diagnosis(vectordb, symptoms):
    llm_diagnosis = OpenAI(temperature=0)
    retriever = vectordb

    diagnosis_template = """Given the following symptoms of a patient known as human and the knowledge given in the 
    Relevant Medical texts, give a 3 possible diagnosis and a number from 0 to 100 with the confidence level you have 
    in the diagnosis. Followed by the medical specialist the patient should visit given the diagnosis. You must give 
    the diagnosis and the specialist in spanish. Give the diagnosis in the following format: 
    
    [{{"diagnosis": "diagnosis1","confidence level": 90,"specialist": "specialist1"}},
    {{"diagnosis": "diagnosis2", "confidence level": 70,"specialist": "specialist2"}},
    {{"diagnosis": "diagnosis3","confidence level": 65, "specialist": "specialist3"}}]
    
    Human symptoms: {question} ----------- Relevant Medical Texts: {context} -----------Give the diagnosis:"""

    DIAGNOSIS_PROMPT = PromptTemplate(template=diagnosis_template, input_variables=["question", "context"])

    diagnosis_model = RetrievalQA.from_chain_type(
        llm=llm_diagnosis,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": DIAGNOSIS_PROMPT})

    return diagnosis_model(symptoms)["result"]


def get_symptoms_summary(chat_history):
    llm = ChatOpenAI(temperature=0)
    prompt_template = PromptTemplate.from_template(
        """Given the chat history provided below, which consists of interactions between a patient (referred to as 
        "Human") and a doctor (referred to as "Assistant"), please summarize the symptoms explicitly mentioned or 
        confirmed by the patient (Human).
        
        *****chat history *****
        {chat} 
        
        """
    )
    prompt_template.format(chat=chat_history)
    summary_model = LLMChain(llm=llm, prompt=prompt_template, output_key="summary")

    return summary_model(chat_history)


def create_medical_report(chat_history):
    llm = ChatOpenAI(temperature=0)
    prompt_report_template = PromptTemplate.from_template(
        """Given the following chat history: 
        *****Chat history between Human and Assistant*****
        {chat}
        *****
        Generate the human medical report in spanish. The report will be reviewed by a doctor. 
        Adhere the report to the structure:

        **Reason for consultation:** - Provide a brief description of why the patient is seeking medical attention or 
        the main reason for the consultation.
        
        **Current Clinical History:** - Describe in detail and chronologically the symptoms and events related to the 
        current illness or condition, from its onset to the time of consultation.
        
        **Diagnosis:**
        - State the main diagnosis based on the gathered information and the tests performed.
        - Mention possible differential diagnoses, if any.
        
        **Tests:**
        - List the possible tests to be performed to confirm the diagnosis."""
    )
    prompt_report_template.format(chat=chat_history)
    report_model = LLMChain(llm=llm, prompt=prompt_report_template, output_key="report")

    return report_model(chat_history)


def main():
    # langchain.debug = True
    # initial session_state in order to avoid refresh
    patient = DetaClient.get_patient(st.session_state.dni)

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
    if "birthday" not in st.session_state:
        st.session_state.birthday = patient["birth_date"]
    if "gender" not in st.session_state:
        st.session_state.gender = patient["sex"]
    if "iterations" not in st.session_state:
        st.session_state.iterations = 0
    if "summary_symptoms" not in st.session_state:
        st.session_state.summary_symptoms = ""
    if "responses" not in st.session_state:
        st.session_state.responses = None

    # Sidebar for personal information
    with st.sidebar:
        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

        profile_pic_path = current_dir / "assets" / f"{st.session_state.name}.png"
        profile_pic = Image.open(profile_pic_path)

        st.sidebar.header("Your Information")
        st.session_state.authenticator.logout('Logout', 'sidebar', key='unique_key')
        st.image(profile_pic, width=250)

        st.markdown(f"""
                        **Nombre**: {st.session_state.name}\n
                        **DNI**: {st.session_state.dni}\n  
                        **Fecha de Nacimiento**: {st.session_state.birthday}\n  
                        **Género**: {st.session_state.gender}\n  
                        **Dirección**: Calle Real No. 45, Ciudad Central\n  
                        **Teléfono**: +34630547119\n
                        """)

        next_appointment = datetime.date(2023, 9, 28)
        next_appointment_2 = datetime.date(2023, 10, 30)

        with st.expander("Próximas Citas", expanded=True):
            st.date_input(label="Rayos X", value=next_appointment, disabled=False, format="DD/MM/YYYY")

        personal_message = f"¡Hola {st.session_state.name}!, estoy aquí para ayudarte a" \
                           f" encontrar al especialista adecuado para tus problemas de salud." \
                           f" ¿Puedes contarme los síntomas que estás experimentando?"

    message(personal_message, is_user=False, avatar_style="big-smile")

    finish_conversation = st.session_state.iterations >= MedicBotConstants.ITERATIONS
    user_question_1 = st.chat_input(placeholder="Describe tus síntomas", disabled=finish_conversation)
    print(user_question_1)
    if user_question_1:
        user_question = translate_english_spanish(user_question_1)['translation']
        if not st.session_state.chat_history:
            pass
        else:
            for i, msg in enumerate(st.session_state.chat_history):
                translated_msg = translate_english_spanish(msg.content)['translation']
                if i % 2 == 0:
                    message(translated_msg, is_user=True, key=str(i) + '_user', avatar_style="personas")
                else:
                    message(translated_msg, is_user=False, key=str(i) + '_ai', avatar_style="big-smile")

        message(user_question_1, is_user=True, key=str(datetime.datetime.now()) + '_user', avatar_style="personas")

        if not finish_conversation:
            with st.spinner("Procesando..."):
                st.session_state.responses = st.session_state.conversation_chain({'question': user_question})
                translated_answer = translate_english_spanish(st.session_state.responses['answer'])['translation']
                message(translated_answer, is_user=False, key=str(datetime.datetime.now()) + '_ai',
                        avatar_style="big-smile")

                st.session_state.chat_history = st.session_state.responses['chat_history']
                # st.session_state.summary_symptoms = get_symptoms_summary(str(st.session_state.chat_history))
                # st.write(st.session_state.summary_symptoms["summary"])

        else:
            with st.status("Buscando tu cita ...", expanded=True) as status:

                st.write('Procesando tus síntomas para encontrar tu especialista...')
                st.session_state.summary_symptoms = get_symptoms_summary(str(st.session_state.chat_history))
                diagnosis = get_diagnosis(st.session_state.vectordb, st.session_state.summary_symptoms["summary"])
                diagnosis_list = json.loads(diagnosis)

                specialist = diagnosis_list[0]["specialist"]
                st.write(f'Te recomiendo que cojas cita con el {specialist}. Buscando en su agenda ...')

                medical_report = create_medical_report(st.session_state.chat_history)["report"]

                DetaClient.update_patient(updates={"diagnosis": diagnosis}, dni=st.session_state.dni)
                DetaClient.update_patient(updates={"report": medical_report}, dni=st.session_state.dni)

                st.date_input(label=f"{specialist}", value=next_appointment_2, disabled=False, format="DD/MM/YYYY")

                status.update(label="Tenemos cita!", state="complete", expanded=True)

                st.button("Coger Cita")

        # number of interactions with the patient
        st.session_state.iterations += 1
