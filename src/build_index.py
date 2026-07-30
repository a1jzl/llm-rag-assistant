"""Construction de l'index vectoriel FAISS a partir des chunks.

Ce module lit les chunks generes par ingest.py, calcule leurs
embeddings avec un modele sentence-transformers, puis persiste
l'index FAISS sur disque pour une utilisation par le retriever.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNKS_PATH = Path("data/chunks.jsonl")
INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/metadata.npy")


def load_chunks(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            _, _, content = line.rstrip("\n").split("\t", 2)
            texts.append(content)
    logger.info("Chunks charges pour indexation: %d", len(texts))
    return texts


def build_faiss_index(texts: list[str], model_name: str = EMBEDDING_MODEL) -> tuple[faiss.Index, np.ndarray]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    logger.info("Index FAISS construit avec %d vecteurs de dimension %d", index.ntotal, dimension)
    return index, embeddings


def main() -> None:
    texts = load_chunks(CHUNKS_PATH)
    index, _ = build_faiss_index(texts)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(METADATA_PATH, np.array(texts, dtype=object))
    logger.info("Index et metadonnees sauvegardes dans %s", INDEX_PATH.parent)


if __name__ == "__main__":
    main()
