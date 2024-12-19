import sqlite3
import faiss
import logging
import numpy as np
import os

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


def insert_metadata(metadata: str):

    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vectors (metadata) VALUES (?)", (metadata,))
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


def test_database_creation():

    setup_database()
    db_file = 'data.db'
    if os.path.exists(db_file):
        print("Banco de dados criado com sucesso!")
    else:
        print("Erro: Banco de dados não foi criado!")


# if __name__ == '__main__':
#     test_database_creation()
