from app import db
from app.models import User, OtherModel

def create_db():
    db.create_all()