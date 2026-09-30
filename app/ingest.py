import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    LOCAL_INDEX_FILE,
    DATA_DIR,
)
from app.vector_store import get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Loads a PDF file and extracts text page by page with 1-based page metadata."""
    path = Path(pdf_path)
    if not path.exists():
        logger.info(f"PDF not found locally at: {pdf_path}. Downloading from official URL...")
        import urllib.request
        download_url = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(download_url, str(path))
            logger.info(f"Downloaded PDF successfully to {path}")
        except Exception as e:
            raise FileNotFoundError(f"PDF file not found at {pdf_path} and download failed: {e}")

    logger.info(f"Loading PDF from: {pdf_path}")
    reader = pypdf.PdfReader(str(path))
    pages_data = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Clean text slightly (remove null bytes or unprintable characters)
        cleaned_text = text.replace("\x00", "").strip()
        if cleaned_text:
            pages_data.append({
                "page": idx + 1,  # 1-indexed page number
                "text": cleaned_text,
            })

    logger.info(f"Extracted {len(pages_data)} non-empty pages from {len(reader.pages)} total pages.")
    return pages_data


def chunk_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits extracted page texts into overlapping chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_counter = 0

    for page_info in pages_data:
        page_num = page_info["page"]
        raw_text = page_info["text"]
        chunks = splitter.split_text(raw_text)

        for text_chunk in chunks:
            text_chunk = text_chunk.strip()
            if len(text_chunk) > 20:  # Skip tiny fragments
                chunk_counter += 1
                all_chunks.append({
                    "id": f"chunk-{chunk_counter}",
                    "text": text_chunk,
                    "page": page_num,
                })

    logger.info(f"Generated {len(all_chunks)} text chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
    return all_chunks


def upsert_to_pinecone(vector_records: List[Dict[str, Any]]) -> bool:
    """Upserts vector records to Pinecone index in batches."""
    if not PINECONE_API_KEY:
        logger.warning("PINECONE_API_KEY is not configured in .env. Skipping Pinecone upload.")
        return False

    try:
        from pinecone import Pinecone, ServerlessSpec

        logger.info(f"Connecting to Pinecone serverless (cloud={PINECONE_CLOUD}, region={PINECONE_REGION})...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pc.list_indexes()

        if hasattr(indexes, "names"):
            existing_names = indexes.names()
        else:
            existing_names = [idx.name if hasattr(idx, "name") else idx.get("name") for idx in indexes]

        if PINECONE_INDEX not in existing_names:
            logger.info(f"Index '{PINECONE_INDEX}' not found. Creating serverless index...")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            logger.info(f"Index '{PINECONE_INDEX}' created successfully.")

        index = pc.Index(PINECONE_INDEX)

        # Batch upsert (batch size = 50)
        batch_size = 50
        total = len(vector_records)
        logger.info(f"Upserting {total} vectors to Pinecone index '{PINECONE_INDEX}'...")

        for i in range(0, total, batch_size):
            batch = vector_records[i : i + batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted vectors {i + 1} to {min(i + batch_size, total)} of {total}")

        logger.info("Pinecone ingestion complete!")
        return True
    except Exception as e:
        logger.error(f"Failed to upsert to Pinecone: {e}")
        return False


def save_local_cache(vector_records: List[Dict[str, Any]]):
    """Saves records locally to ensure high reliability and offline testability."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(vector_records, f, indent=2)
    logger.info(f"Saved {len(vector_records)} vectors to local vector cache: {LOCAL_INDEX_FILE}")


def run_ingestion():
    """Main ingestion pipeline."""
    print("=" * 60)
    print(" RAG Chatbot: PDF Ingestion Pipeline")
    print("=" * 60)

    # 1. Load PDF
    pages = load_pdf(PDF_PATH)

    # 2. Chunk text
    chunks = chunk_pages(pages)

    # 3. Generate embeddings
    embedder = get_embedder()
    logger.info("Generating embeddings for all chunks...")
    texts_to_embed = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts_to_embed)

    vector_records = []
    for chunk, emb in zip(chunks, embeddings):
        vector_records.append({
            "id": chunk["id"],
            "values": emb,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_id": chunk["id"],
            },
        })

    # 4. Save local backup cache
    save_local_cache(vector_records)

    # 5. Upsert to Pinecone if configured
    pinecone_success = upsert_to_pinecone(vector_records)

    print("=" * 60)
    print(f" Ingestion Summary:")
    print(f" - Non-empty Pages: {len(pages)}")
    print(f" - Total Chunks: {len(chunks)}")
    print(f" - Embeddings Dimension: {EMBEDDING_DIMENSION}")
    print(f" - Local Cache: {LOCAL_INDEX_FILE} (Ready)")
    print(f" - Pinecone Upload: {'SUCCESS' if pinecone_success else 'PENDING / SKIPPED (Check PINECONE_API_KEY)'}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
