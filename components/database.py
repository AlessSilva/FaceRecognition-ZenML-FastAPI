import sqlite3
import faiss
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_database(
):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS vectors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metadata TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()


def insert_metadata(
    metadata: str
) -> int:
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vectors VALUES (?)",
                   (metadata,))
    conn.commit()
    vector_id = cursor.lastrowid
    conn.close()
    return vector_id


def setup_faiss_index(
    dimension
):
    index = faiss.IndexFlatL2(dimension)
    return index


def add_vector_to_index(
    index,
    vector,
    metadata
):
    vector_id = insert_metadata(metadata)
    vector = np.array([vector],
                      dtype='float32')
    index.add(vector)
    logger.info(f"Vector added to FAISS with metadata: {metadata}")
    return vector_id


def search_similar_vectors(
    index,
    query_vector,
    top_k=5
):
    query_vector = np.array([query_vector], dtype='float32')
    distances, indices = index.search(query_vector, top_k)
    return distances, indices
