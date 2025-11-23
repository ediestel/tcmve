#!/usr/bin/env python3
"""
Query game narratives using embeddings for semantic similarity search.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
import psycopg2
from dotenv import load_dotenv
import openai
import tiktoken

load_dotenv()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def count_tokens(text: str, model: str = "text-embedding-3-large") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate
        return len(text.split()) * 4 // 3

def chunk_text_with_overlap(text: str, chunk_size: int = 8000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)  # GPT-4 encoding

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tiktoken.get_encoding("cl100k_base").decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move start position with overlap
        start = end - overlap
        if start >= len(tokens):
            break

    return chunks

def embed_text_direct(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Generate embedding directly without chunking."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def embed_long_text(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Embed long text using chunking with overlap and averaging."""
    token_count = count_tokens(text, model)

    if token_count <= 8000:
        # No chunking needed
        return embed_text_direct(text, model)

    # Chunk and average embeddings
    chunks = chunk_text_with_overlap(text, chunk_size=8000, overlap=200)
    embeddings = []

    for chunk in chunks:
        embedding = embed_text_direct(chunk, model)
        embeddings.append(embedding)

    # Average the embeddings
    return np.mean(embeddings, axis=0).tolist()

def embed_query(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Generate embedding for query text with automatic chunking."""
    return embed_long_text(text, model)

def find_similar_games(query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
    """
    Find games with narratives most similar to the query.

    Returns: List of (game_name, similarity_score, narrative_preview)
    """
    query_embedding = embed_query(query)

    conn = None
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

        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT game_name, embedding, narrative_text
                FROM game_narratives
                ORDER BY game_name
            """)

            results = []
            for row in cursor.fetchall():
                game_name, embedding_json, narrative_text = row
                embedding = json.loads(embedding_json)
                similarity = cosine_similarity(query_embedding, embedding)
                preview = narrative_text[:200] + "..." if len(narrative_text) > 200 else narrative_text
                results.append((game_name, similarity, preview))

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    except Exception as e:
        print(f"Error querying embeddings: {e}")
        return []
    finally:
        if conn:
            conn.close()

def find_similar_chunks(query: str, document_id: str = None, top_k: int = 5) -> List[Tuple[str, int, float, str]]:
    """
    Find document chunks most similar to the query.

    Args:
        query: Search query
        document_id: Optional - filter by specific document
        top_k: Number of results to return

    Returns: List of (document_id, chunk_index, similarity_score, chunk_text)
    """
    query_embedding = embed_query(query)

    conn = None
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

        with conn.cursor() as cursor:
            if document_id:
                cursor.execute("""
                    SELECT document_id, chunk_index, embedding, chunk_text
                    FROM document_chunks
                    WHERE document_id = %s
                    ORDER BY chunk_index
                """, (document_id,))
            else:
                cursor.execute("""
                    SELECT document_id, chunk_index, embedding, chunk_text
                    FROM document_chunks
                    ORDER BY document_id, chunk_index
                """)

            results = []
            for row in cursor.fetchall():
                doc_id, chunk_idx, embedding_json, chunk_text = row
                embedding = json.loads(embedding_json)
                similarity = cosine_similarity(query_embedding, embedding)
                results.append((doc_id, chunk_idx, similarity, chunk_text))

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]

    except Exception as e:
        print(f"Error querying chunk embeddings: {e}")
        return []
    finally:
        if conn:
            conn.close()

def main():
    """Demo: Find similar games for sample queries including long text."""
    sample_queries = [
        "cooperation vs competition in moral dilemmas",
        "trust and betrayal in relationships",
        "strategic decision making under uncertainty",
        "evolution of cooperation in society",
        # Long query to test chunking (>1000 tokens)
        """In the context of Thomistic philosophy and virtue ethics, how do different game theory scenarios illuminate the tension between individual self-interest and the common good? Consider the Prisoner's Dilemma where mutual cooperation represents participation in divine unity (unum) while defection mirrors the fallen state of disconnected beings. How does the Ultimatum Game demonstrate justice (justitia) through fair distribution, and what does the Stag Hunt reveal about the virtue of fortitude in trusting collective action over solitary safety? Furthermore, analyze how evolutionary game theory might model the gradual perfection of virtues through repeated interactions, where initial self-serving strategies give way to cooperative habits that align with teleological ends. What role do Nash equilibria play in understanding the 'natural law' of rational choice, and how might divine providence influence the emergence of cooperative outcomes in iterated games? Finally, consider whether these mathematical models can truly capture the transcendent aspects of human freedom and grace in moral decision-making."""
    ]

    for i, query in enumerate(sample_queries):
        token_count = count_tokens(query)
        chunking_needed = " (will use chunking)" if token_count > 1000 else ""

        print(f"\nQuery {i+1}: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"Tokens: {token_count}{chunking_needed}")
        print("-" * 50)

        similar_games = find_similar_games(query, top_k=2)
        for game_name, score, preview in similar_games:
            print(".3f")
            print(f"Preview: {preview}")
            print()

if __name__ == "__main__":
    main()