import os
from constants import DETA_KEY
# from dotenv import load_dotenv
from deta import Deta

# load_dotenv(".env")
# Initialize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("Mapfre_Base")


# insert
def insert_patient_info(dni, first_name, last_name, age):
    """Returns the report on a successful creation, otherwise raises an error"""
    return db.insert({"key": dni, "first_name": first_name, "last_name": last_name, "age": age})


def fetch_all_patients():
    """Returns a dict with all patients"""
    res = db.fetch()
    return res.items


def get_patient(dni):
    """If not found, the function will return None"""
    return db.get(dni)


def update_patient(updates, dni):
    """If the item is updated, returns None. Otherwise, an exception is raised
       Example of use: update_user(updates={"surname": Gonzalez, "password":hfi~~€enf1}, key=dni)"""
    return db.update(updates, dni)

