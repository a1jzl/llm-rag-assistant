"""Recherche semantique dans l'index FAISS.

Ce module encapsule la logique de recherche des k chunks les plus
pertinents par rapport a une requete utilisateur, en s'appuyant sur
le meme modele d'embedding que celui utilise lors de l'indexation.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/metadata.npy")
TOP_K = 4


class Retriever:
    """Recherche les chunks les plus pertinents pour une requete donnee."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.metadata = np.load(METADATA_PATH, allow_pickle=True)

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        """Retourne les top_k chunks les plus proches semantiquement de query."""
        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = [self.metadata[i] for i in indices[0] if i != -1]
        logger.info("Requete: %s | %d resultats trouves", query, len(results))
        return results


if __name__ == "__main__":
    retriever = Retriever()
    hits = retriever.search("Quelle est la procedure de remboursement ?")
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] {hit[:200]}")
