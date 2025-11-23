import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        conn = psycopg2.connect(database_url)
    else:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "llm_exam_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
    print("Database connection successful")
    conn.close()
except Exception as e:
    print(f"Database connection failed: {e}")