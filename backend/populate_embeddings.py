#!/usr/bin/env python3
"""
Populate game_narratives table with embeddings for all game narratives.
"""

import os
import hashlib
import json
from typing import Dict, List
import psycopg2
from dotenv import load_dotenv
import openai

load_dotenv()

def get_game_narratives() -> Dict[str, str]:
    """Extract narratives from all game modules."""
    import importlib
    import sys
    import os

    # Add games directory to path
    games_dir = os.path.join(os.path.dirname(__file__), 'games')
    if games_dir not in sys.path:
        sys.path.insert(0, games_dir)

    narratives = {}

    # List of game modules
    game_modules = [
        'auction', 'chicken', 'evolution', 'multiplay', 'prisoner',
        'regret', 'repeated_pd', 'shadow', 'stackelberg', 'stag_hunt', 'ultimatum'
    ]

    for module_name in game_modules:
        try:
            module = importlib.import_module(module_name)
            narrative = module.__doc__.strip() if module.__doc__ else ""
            if narrative:
                narratives[module_name] = narrative
        except Exception as e:
            print(f"Error loading {module_name}: {e}")

    return narratives

def generate_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Generate embedding for text using OpenAI."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def populate_narratives():
    """Populate the game_narratives table with embeddings."""
    narratives = get_game_narratives()

    if not narratives:
        print("No narratives found!")
        return

    conn = None
    try:
        # Use DATABASE_URL if available, otherwise construct from components
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

        with conn.cursor() as cursor:
            print(f"Populating embeddings for {len(narratives)} games...")

            for game_name, narrative_text in narratives.items():
                # Create hash for deduplication
                text_hash = hashlib.sha256(narrative_text.encode()).hexdigest()

                # Check if already exists
                cursor.execute(
                    "SELECT id FROM game_narratives WHERE text_hash = %s",
                    (text_hash,)
                )
                if cursor.fetchone():
                    print(f"✓ {game_name} already exists, skipping")
                    continue

                # Generate embedding
                print(f"Generating embedding for {game_name}...")
                embedding = generate_embedding(narrative_text)
                embedding_model = "text-embedding-3-large"

                # Insert into database
                cursor.execute("""
                    INSERT INTO game_narratives
                    (game_name, narrative_text, text_hash, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    game_name,
                    narrative_text,
                    text_hash,
                    json.dumps(embedding),
                    embedding_model
                ))

                print(f"✓ Inserted {game_name}")

            conn.commit()
            print("\n🎉 All game narratives populated with embeddings!")

    except Exception as e:
        print(f"❌ Error populating narratives: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("TCMVE Game Narratives Embedding Population")
    print("=" * 50)
    populate_narratives()