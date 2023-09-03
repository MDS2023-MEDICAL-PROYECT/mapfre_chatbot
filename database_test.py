import calendar  # Core Python Module
from datetime import datetime  # Core Python Module
import streamlit as st  # pip install streamlit
import database as db

# -------------- SETTINGS --------------

page_title = "Insert Patient Info"
page_icon = "🤧"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
# --------------------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)


def main():
    tab1, tab2 = st.tabs(["Insert", "Fetch"])
    with tab1:
        if 'name' not in st.session_state:
            st.session_state.name = ''
        if 'dni' not in st.session_state:
            st.session_state.dni = ''
        if 'surname' not in st.session_state:
            st.session_state.surname = ''
        if 'age' not in st.session_state:
            st.session_state.age = 0

        with st.form("my_form", clear_on_submit=True):
            dni = st.text_input(key="input_dni", label="DNI:", value=st.session_state.dni)
            name = st.text_input(key="input_name", label="Nombre:", value=st.session_state.name)
            surname = st.text_input(key="input_surname", label="Apellidos:", value=st.session_state.surname)
            age = st.number_input(key="input_age", label="Edad:", min_value=0, max_value=120, value=st.session_state.age)

            save_button = st.form_submit_button("Guardar")

        if save_button:
            try:
                db.insert_patient_info(dni, name, surname, age)

                st.session_state.name = ''
                st.session_state.dni = ''
                st.session_state.surname = ''
                st.session_state.age = 0

                st.success('Patient inserted successfully', icon="✅")
            except:
                st.warning('Patient already exist', icon="⚠️")

    with tab2:
        if 'name' not in st.session_state:
            st.session_state.name = ''
        if 'surname' not in st.session_state:
            st.session_state.surname = ''
        if 'age' not in st.session_state:
            st.session_state.age = -1
        if 'patient' not in st.session_state:
            st.session_state.patient = None

        dni = st.text_input(key="load_dni", label="DNI:")

        load_button = st.button("Load")

        if load_button:

            st.session_state.patient = db.get_patient(dni)

            if st.session_state.patient is None:
                st.warning('Patient not found', icon="⚠️")
            else:
                st.session_state.name = st.session_state.patient['first_name']
                st.session_state.surname = st.session_state.patient['last_name']
                st.session_state.age = st.session_state.patient['age']

                with st.container():
                    st.write(st.session_state.name)
                    st.write(st.session_state.surname)
                    st.write(st.session_state.age)


if __name__ == "__main__":
    main()
