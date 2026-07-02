from app.database.db import SessionLocal
from app.database.models import Application


class ImportService:

    def __init__(self):
        self.db = SessionLocal()

    def application_exists(self, application_id):

        return self.db.query(Application).filter(
            Application.application_id == application_id
        ).first()

    def save(self, application):

        self.db.add(application)

    def commit(self):

        self.db.commit()

    def close(self):

        self.db.close()