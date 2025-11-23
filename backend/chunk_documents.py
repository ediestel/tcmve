#!/usr/bin/env python3
"""
Chunk long documents and store embeddings for semantic search.
Supports PDFs, text files, and other document formats.
"""

import os
import hashlib
import json
import fitz  # PyMuPDF for PDF processing
from typing import List, Dict, Any, Optional
import psycopg2
from dotenv import load_dotenv
import openai
import tiktoken

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Split text into chunks with metadata."""
    tokens = tiktoken.get_encoding("cl100k_base").encode(text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tiktoken.get_encoding("cl100k_base").decode(chunk_tokens)

        chunks.append({
            'chunk_index': chunk_index,
            'chunk_text': chunk_text,
            'start_position': start,
            'end_position': end,
            'token_count': len(chunk_tokens)
        })

        chunk_index += 1
        start = end - overlap
        if start >= len(tokens):
            break

    return chunks

def generate_chunk_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Generate embedding for a chunk."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def store_document_chunks(document_id: str, document_title: str, chunks: List[Dict[str, Any]],
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """Store document chunks with embeddings in database."""
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
            print(f"Storing {len(chunks)} chunks for document: {document_title}")

            for chunk in chunks:
                text_hash = hashlib.sha256(chunk['chunk_text'].encode()).hexdigest()

                # Check if chunk already exists
                cursor.execute(
                    "SELECT id FROM document_chunks WHERE text_hash = %s",
                    (text_hash,)
                )
                if cursor.fetchone():
                    print(f"✓ Chunk {chunk['chunk_index']} already exists, skipping")
                    continue

                # Generate embedding
                print(f"Generating embedding for chunk {chunk['chunk_index']}...")
                embedding = generate_chunk_embedding(chunk['chunk_text'])

                # Store chunk
                cursor.execute("""
                    INSERT INTO document_chunks
                    (document_id, document_title, chunk_index, chunk_text, text_hash,
                     embedding, embedding_model, start_position, end_position, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    document_id,
                    document_title,
                    chunk['chunk_index'],
                    chunk['chunk_text'],
                    text_hash,
                    json.dumps(embedding),
                    "text-embedding-3-large",
                    chunk['start_position'],
                    chunk['end_position'],
                    json.dumps(metadata or {})
                ))

            conn.commit()
            print(f"✓ Stored all chunks for {document_title}")

    except Exception as e:
        print(f"❌ Error storing chunks: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def process_document(file_path: str, document_id: str, title: str = None,
                    chunk_size: int = 1000, overlap: int = 200) -> None:
    """Process a document file and store its chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Extract text based on file type
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if not text.strip():
        raise ValueError("No text content found in document")

    # Generate chunks
    chunks = chunk_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)

    # Metadata
    metadata = {
        'source_file': file_path,
        'total_tokens': sum(c['token_count'] for c in chunks),
        'num_chunks': len(chunks),
        'chunk_size': chunk_size,
        'overlap': overlap
    }

    # Store chunks
    store_document_chunks(document_id, title or os.path.basename(file_path), chunks, metadata)

def main():
    """Example usage."""
    # Example: Process a PDF
    # process_document(
    #     file_path="/path/to/thomistic_text.pdf",
    #     document_id="summa_theologica_vol1",
    #     title="Summa Theologica - Volume 1",
    #     chunk_size=1000,
    #     overlap=200
    # )

    print("Document chunking system ready.")
    print("Use process_document() to chunk and store document embeddings.")

if __name__ == "__main__":
    main()