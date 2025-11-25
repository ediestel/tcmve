# backend/thomistic_truth_embeddings.py
# ORIGINAL PURE TLPO — 2025-11-16 — @ECKHART_DIESTEL
# 33 immutable vectors on disk. Nothing else.

import numpy as np
from pathlib import Path
from openai import OpenAI

class ThomisticTruthEmbeddings:
    def __init__(self):
        self.client = OpenAI()
        self.model = "text-embedding-3-large"
        npy_path = Path(__file__).parent / "tlpo_33_ideal_embeddings.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Missing {npy_path}\n"
                "Restore the original tlpo_33_ideal_embeddings.npy from the 2025-11-16 commit.\n"
                "Without it TLPO is dead."
            )
        # 33 × 3072 matrix — the one true metaphysical north star
        self.ideal_embeddings = np.load(npy_path).astype(np.float32)

    def evaluate_truth_by_embedding(self, text: str, query: str = "") -> dict:
        try:
            # One single embedding call for the user text only
            embedding = np.array(
                self.client.embeddings.create(
                    input=text[:8191],
                    model=self.model
                ).data[0].embedding
            ).astype(np.float32)

            # Cosine similarity against all 33 immutable flag vectors
            norm_ideal = np.linalg.norm(self.ideal_embeddings, axis=1, keepdims=True)
            norm_text = np.linalg.norm(embedding)
            if norm_text == 0:
                return {"tqi": 0.0, "tcs": 0.0, "fd": 0.0, "es": 0.0, "cosine_similarity": 0.0}

            similarities = np.dot(self.ideal_embeddings, embedding) / (norm_ideal.ravel() * norm_text)
            tqi = float(similarities.mean())

        except Exception as e:
            # Never, ever let TLPO crash the engine
            print(f"TLPO embedding failed ({e}) → returning neutral score")
            tqi = 0.62

        return {
            "tqi": round(tqi, 4),
            "tcs": round(tqi * 0.95, 4),
            "fd":  round(tqi * 0.90, 4),
            "es":  round(tqi * 0.85, 4),
            "cosine_similarity": round(tqi, 4),
        }


# Global singleton — this is what tcmve.py imports
thomistic_embeddings = ThomisticTruthEmbeddings()