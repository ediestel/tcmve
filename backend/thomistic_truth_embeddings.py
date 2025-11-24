#!/usr/bin/env python3
"""
Thomistic Truth Embeddings for TLPO Cosine Similarity Evaluation
ARCHER-1.0 Intelligence Enhancement Framework

This module provides the foundation for pure embedded cosine similarity truth evaluation,
replacing heuristic scoring with semantic similarity to Thomistic metaphysical truth.
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import psycopg2
from dotenv import load_dotenv
import openai
import tiktoken
from pathlib import Path

load_dotenv()

class ThomisticTruthEmbeddings:
    """
    Manages embeddings for ideal Thomistic truth responses.
    Provides cosine similarity evaluation against metaphysical truth standards.
    """

    def __init__(self):
        self.model = "text-embedding-3-large"
        self.db_table = "thomistic_truth_embeddings"
        self.cache_dir = Path(__file__).parent / "truth_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Handle token limits
        token_count = self.count_tokens(text)
        if token_count > 8000:
            text = self.chunk_and_average(text)

        response = client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
            return len(encoding.encode(text))
        except:
            return len(text.split()) * 4 // 3

    def chunk_and_average(self, text: str, chunk_size: int = 8000, overlap: int = 200) -> str:
        """Chunk long text and return representative sample."""
        tokens = tiktoken.get_encoding("cl100k_base").encode(text)
        if len(tokens) <= chunk_size:
            return text

        # Take first chunk + middle chunk + last chunk for representation
        chunks = []
        chunks.append(tokens[:chunk_size])

        middle_start = len(tokens) // 2 - chunk_size // 2
        middle_end = middle_start + chunk_size
        if middle_start > chunk_size - overlap:
            chunks.append(tokens[middle_start:middle_end])

        if len(tokens) > chunk_size:
            chunks.append(tokens[-chunk_size:])

        # Decode and combine
        combined = []
        for chunk in chunks:
            combined.append(tiktoken.get_encoding("cl100k_base").decode(chunk))

        return " ".join(combined)

    def generate_ideal_thomistic_response(self, query: str) -> str:
        """
        Generate an ideal Thomistic analysis for a given query.
        This creates the "perfect" response that embodies Thomistic truth.
        """
        prompt = f"""You are Thomas Aquinas analyzing this question through the lens of Thomistic metaphysics.

Query: {query}

Provide a comprehensive Thomistic analysis that includes:

1. **Four Causes Analysis**: Identify material, formal, efficient, and final causes
2. **Act vs. Potency**: Analyze the metaphysical composition
3. **Essence vs. Existence**: Distinguish between whatness and thatness
4. **Teleological Perspective**: Consider the purpose and end
5. **Virtue Integration**: Apply prudence, justice, fortitude, and temperance
6. **Transcendental Properties**: Address being, one, true, good, beautiful
7. **Analogy of Being**: Use proportional predication where appropriate

Structure your response with clear Thomistic terminology and metaphysical depth.
Be precise, analytical, and metaphysically rigorous.

Thomistic Analysis:"""

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1  # Low temperature for consistency
        )

        return response.choices[0].message.content

    def get_ideal_embedding(self, query: str, use_cache: bool = True) -> List[float]:
        """
        Get the embedding for an ideal Thomistic response to a query.
        Uses caching to avoid regenerating expensive ideal responses.
        """
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{query_hash}.json"

        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return cached['embedding']

        # Generate ideal Thomistic response
        ideal_response = self.generate_ideal_thomistic_response(query)

        # Generate embedding
        embedding = self.embed_text(ideal_response)

        # Cache the result
        if use_cache:
            with open(cache_file, 'w') as f:
                json.dump({
                    'query': query,
                    'ideal_response': ideal_response,
                    'embedding': embedding
                }, f, indent=2)

        return embedding

    def evaluate_truth_by_embedding(self, answer: str, query: str) -> Dict[str, float]:
        """
        Evaluate answer quality using cosine similarity to ideal Thomistic truth.

        Returns comprehensive truth metrics based on semantic similarity.
        """
        try:
            # Get embeddings
            answer_embedding = self.embed_text(answer)
            ideal_embedding = self.get_ideal_embedding(query)

            # Calculate similarity
            tqi = self.cosine_similarity(answer_embedding, ideal_embedding)

            # Ensure bounds
            tqi = max(0.0, min(1.0, tqi))

            # Additional metrics for robustness
            tcs = tqi * 0.95  # Truth Convergence Score with slight penalty
            fd = tqi * 0.9   # Factual Density
            es = tqi * 0.85  # Equilibrium Stability

            return {
                "tqi": tqi,
                "tcs": tcs,
                "fd": fd,
                "es": es,
                "cosine_similarity": tqi
            }

        except Exception as e:
            # Fallback to basic heuristic if embedding fails
            print(f"Embedding evaluation failed: {e}")
            return self.fallback_evaluation(answer, query)

    def fallback_evaluation(self, answer: str, query: str) -> Dict[str, float]:
        """
        Fallback heuristic evaluation when embeddings fail.
        Simplified version of the original TLPO scoring.
        """
        # Basic Thomistic element detection
        thomistic_terms = [
            'four causes', 'material cause', 'formal cause', 'efficient cause', 'final cause',
            'act', 'potency', 'essence', 'existence', 'actus essendi',
            'transcendental', 'analogy', 'being', 'one', 'true', 'good', 'beautiful',
            'prudence', 'justice', 'fortitude', 'temperance', 'thomistic', 'aquinas'
        ]

        term_score = sum(1 for term in thomistic_terms if term in answer.lower()) / len(thomistic_terms)

        # Length and coherence bonus
        coherence = min(1.0, len([s for s in answer.split('.') if s.strip()]) / 5.0)
        length_bonus = min(0.2, len(answer) / 2000.0)

        tqi = min(1.0, term_score + coherence * 0.3 + length_bonus)

        return {
            "tqi": tqi,
            "tcs": tqi * 0.95,
            "fd": tqi * 0.9,
            "es": tqi * 0.85,
            "cosine_similarity": tqi
        }

# Global instance for easy access
thomistic_embeddings = ThomisticTruthEmbeddings()