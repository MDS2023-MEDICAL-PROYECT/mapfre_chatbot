import streamlit as st
from pathlib import Path
from PIL import Image
import json
from src.clients.database import DetaClient

st.set_page_config(
    layout="wide"
)
st.header("Patient Health Profile")

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

dni = "45101711E"
patient = DetaClient.get_patient(dni)
patient_name = patient["first_name"]
patient_birth = patient["birth_date"]
patient_sex = patient["sex"]
patient_report = patient["report"]
patient_diagnosis = patient["diagnosis"]
# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
profile_pic = current_dir / "assets" / f"{patient_name}.png"

# --- LOAD PROFILE PIC ---
profile_pic = Image.open(profile_pic)


def main():
    with st.container():
        left_col, right_col = st.columns([2, 5])

        # ----- LEFT COLUMN -----
        with right_col:
            with st.expander("Información del Paciente", expanded=True):
                col1, col2 = st.columns([1, 5], gap="small")
                col1.image(profile_pic, width=200)
                col2.markdown(f"""
                **Nombre**: {patient_name}            
                **Fecha de Nacimiento**: {patient_birth}  
                **Sexo**: {patient_sex}  
                **Teléfono**: +52-555-1234567
                """)

            with st.expander("Historia y Antecedentes", expanded=True):
                st.markdown(f"""
                **Antecedentes Médicos**:
                - :heartbeat: **Enfermedades Crónicas**: Hipertensión diagnosticada en 2018.
                - :no_entry_sign: **Alergias**: Penicilina.
                - :hospital: **Cirugías Previas**: Apendicectomía en 1995.

                **Hábitos**:
                - :smoking: **Tabaco**: Fumador activo.
                - :wine_glass: **Alcohol**: Consumo ocasional.

                **Familiares**:
                - :family: Padre fallecido por enfermedad cardiovascular. Madre con diabetes tipo 2.
                
                
                **Historia Actual**: {patient_report}


                """)

        # ----- RIGHT COLUMN -----
        with left_col:
            st.subheader("Posibles Diagnósticos")
            diagnosis_list = json.loads(patient_diagnosis)
            print(diagnosis_list)
            # diagnosis_data = [
            #    ("Litiasis Renal", 70),
            #    ("Apendicitis Crónica", 20),
            #    ("Infección Urinaria", 10)
            # ]

            for diag in diagnosis_list:
                st.markdown(f"**Diagnóstico**: {diag['diagnosis']}")
                st.progress(diag['confidence level'])

                # st.markdown(f"**{diagnosis}**")
                # st.progress(probability)

            # with st.expander("Análisis médico", expanded=True):
            st.text_area(label='Diagnóstico médico', height=400, placeholder='Incluya sus conclusiones')


main()
