from pathlib import Path
import os
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY is not set")

JWT_ALGORITHM = os.getenv('JWT_ALGORITHM')
if not JWT_ALGORITHM:
    raise ValueError("JWT_ALGORITHM is not set")

JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES'))
if not JWT_ACCESS_TOKEN_EXPIRE_MINUTES:
    raise ValueError("JWT_ACCESS_TOKEN_EXPIRE_MINUTES is not set")

DB_USER = os.getenv('DB_USER')
if not DB_USER:
    raise ValueError("DB_USER is not set")

DB_PASSWORD = os.getenv('DB_PASSWORD')
if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD is not set")

DB_HOST = os.getenv('DB_HOST')
if not DB_HOST:
    raise ValueError("DB_HOST is not set")

DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
USE_SQLITE = os.getenv('USE_SQLITE', 'false').lower() == 'true' 