import streamlit as st
from pathlib import Path
from PIL import Image

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
profile_pic = current_dir / "assets" / "sofia_palenciano.png"

# --- LOAD PROFILE PIC ---
profile_pic = Image.open(profile_pic)


def main():

    with st.container():
        left_col, right_col = st.columns([3, 1])

        # ----- LEFT COLUMN -----
        with left_col:
            with st.expander("Información del Paciente", expanded=True):
                col1, col2 = st.columns(2)
                col1.image(profile_pic, width=250)
                col2.markdown("""
                **Nombre**: Sofia Palenciano Triay  
                **Fecha de Nacimiento**: 15 de marzo de 1975  
                **Sexo**: Femenino  
                **Dirección**: Calle Real No. 45, Ciudad Central  
                **Teléfono**: +52-555-1234567
                """)

            with st.expander("Historia y Antecedentes", expanded=True):
                st.markdown("""
                **Historia Actual**: Sofia consulta por episodios recurrentes de dolor abdominal.

                **Antecedentes Médicos**:
                - :heartbeat: **Enfermedades Crónicas**: Hipertensión diagnosticada en 2018.
                - :no_entry_sign: **Alergias**: Penicilina.
                - :hospital: **Cirugías Previas**: Apendicectomía en 1995.

                **Hábitos**:
                - :smoking: **Tabaco**: Fumador activo.
                - :wine_glass: **Alcohol**: Consumo ocasional.

                **Familiares**:
                - :family: Padre fallecido por enfermedad cardiovascular. Madre con diabetes tipo 2.
                """)

        # ----- RIGHT COLUMN -----
        with right_col:
            st.subheader("Posibles Diagnósticos")
            diagnosis_data = [
                ("Litiasis Renal", 70),
                ("Apendicitis Crónica", 20),
                ("Infección Urinaria", 10)
            ]

            for diagnosis, probability in diagnosis_data:
                st.markdown(f"**{diagnosis}**")
                st.progress(probability)

    with st.expander("Chat con el paciente", expanded=False):
        # Initialize a session variable to store chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # If 'user_input' doesn't exist in session state, initialize it
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""

        # If 'message_sent' doesn't exist in session state, initialize it
        if 'message_sent' not in st.session_state:
            st.session_state.message_sent = False

        # If message was sent, render an empty input box
        if st.session_state.message_sent:
            user_text = st.text_input("Tú:", value="")
            if user_text:  # if user has entered something new in the input
                st.session_state.message_sent = False  # reset the flag
        else:
            user_text = st.text_input("Tú:", value=st.session_state.user_input)


main()