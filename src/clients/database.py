
from src.constants.medic_bot import MedicBotConstants
from deta import Deta

# load_dotenv(".env")
# Initialize with a project key
deta = Deta(MedicBotConstants.DETA_KEY)

# This is how to create/connect a database
db = deta.Base("Mapfre_Base")

class DetaClient:
    @classmethod
    def insert_patient_info(cls, dni, first_name, last_name, age):
        """Returns the report on a successful creation, otherwise raises an error"""
        return db.insert({"key": dni, "first_name": first_name, "last_name": last_name, "age": age})

    @classmethod
    def fetch_all_patients(cls):
        """Returns a dict with all patients"""
        res = db.fetch()
        return res.items

    @classmethod
    def get_patient(cls, dni):
        """If not found, the function will return None"""
        return db.get(dni)

    @classmethod
    def update_patient(cls, updates, dni):
        """If the item is updated, returns None. Otherwise, an exception is raised
           Example of use: update_user(updates={"surname": Gonzalez, "password":hfi~~€enf1}, key=dni)"""
        return db.update(updates, dni)
