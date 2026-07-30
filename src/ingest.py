"""Ingestion des documents sources pour le pipeline RAG.

Ce module charge les documents bruts (PDF, Markdown, HTML) depuis un
dossier d'entree, les nettoie, puis les decoupe en chunks avec
chevauchement afin de preparer l'etape d'indexation vectorielle.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


@dataclass
class Chunk:
    """Represente un fragment de document pret a etre embedde."""

    source: str
    content: str
    chunk_id: int


def load_documents(input_dir: Path) -> list:
    """Charge tous les documents PDF, Markdown et HTML d'un dossier."""
    loaders = [
        DirectoryLoader(str(input_dir), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(input_dir), glob="**/*.html", loader_cls=UnstructuredHTMLLoader),
    ]
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as exc:
            logger.warning("Erreur de chargement avec %s: %s", loader, exc)
    logger.info("Documents charges: %d", len(documents))
    return documents


def split_documents(documents: list) -> list[Chunk]:
    """Decoupe les documents en chunks avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = splitter.split_text(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(source=source, content=piece, chunk_id=i))
    logger.info("Chunks generes: %d", len(chunks))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion des documents pour le RAG")
    parser.add_argument("--input", type=str, required=True, help="Dossier des documents source")
    parser.add_argument("--output", type=str, default="data/chunks.jsonl", help="Fichier de sortie")
    args = parser.parse_args()

    input_dir = Path(args.input)
    documents = load_documents(input_dir)
    chunks = split_documents(documents)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"{chunk.source}\t{chunk.chunk_id}\t{chunk.content}\n")

    logger.info("Ingestion terminee. Resultat ecrit dans %s", output_path)


if __name__ == "__main__":
    main()
