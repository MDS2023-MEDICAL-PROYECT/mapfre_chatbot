import streamlit as st
from pathlib import Path
from PIL import Image
import textwrap
import json
from src.clients.database import DetaClient

st.set_page_config(
    layout="wide"
)
st.header("Expediente Sanitario del Paciente")

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
Name = patient["first_name"]
patient_birth = patient["birth_date"]
Gender = patient["sex"]
patient_report = patient["report"]
patient_diagnosis = patient["diagnosis"]

# --- FORMAT REPORT ---
format_patient_report = patient_report.replace("[", "{").replace("]", "}")

# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
profile_pic = current_dir / "assets" / f"{Name}.png"

# --- LOAD PROFILE PIC ---
profile_pic = Image.open(profile_pic)


def main():
    with st.container():
        left_col, center_col, right_col = st.columns(3)

        with left_col:
            with st.expander("", expanded=True):
                col1, col2 = st.columns(2, gap="small")
                col1.image(profile_pic, width=200)
                col2.markdown(f"""
                                **Nombre**: {Name}

                                **Fecha de nacimiento**: {patient_birth}

                                **Género**: {Gender}

                                **Teléfono**: +34653479511
                                """)

                st.markdown("""**Historial Médico**:""")
                st.markdown("""- :heartbeat: **Enfermedades Crónicas**: Hipertensión diagnosticada en 2018.""")
                st.markdown("""- :no_entry_sign: **Alergias**: Penicilina.""")
                st.markdown("""- :hospital: **Cirugías Previas**: Apendicectomía en 1995.""")
                st.markdown("""**Hábitos**:""")
                st.markdown("""- :smoking: **Tabaco**: Fumador activo.""")
                st.markdown("""- :wine_glass: **Alcohol**: Consumo ocasional.""")
                st.markdown("""**Familia**:""")
                st.markdown("""- :family: Padre fallecido por enfermedad cardiovascular. Madre con diabetes tipo 2.""")

        with center_col:
            # st.subheader("Top 3 Diagnosis")
            diagnosis_list = json.loads(patient_diagnosis)

            for diag in diagnosis_list:
                st.markdown(f"**Diagnóstico**: {diag['diagnosis']}({diag['confidence level']})")
                st.progress(diag['confidence level'])

            st.text_area(label='Seguimiento del doctor', height=400, placeholder='Añada sus comentarios')

        with right_col:
            st.markdown("""**Anamnesis Proxima**:""")
            st.write(format_patient_report.format(Name=Name, Age=40, Gender=Gender))

main()
