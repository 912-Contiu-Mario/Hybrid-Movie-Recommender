from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config.environment import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, USE_SQLITE

class Base(DeclarativeBase):
    pass

if USE_SQLITE:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./movie_recommender.db"
    connect_args = {"check_same_thread": True}
else:
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    connect_args = {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

