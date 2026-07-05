from langchain_ollama import OllamaEmbeddings
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from src.config import config
import time
import uuid
import json
from pprint import pprint

class VectorDB:
    def __init__(self, embedding_model:str = config["EMBEDDING_MODEL"], host:str = config["VECTOR_DB_HOST"], port:int = config["VECTOR_DB_PORT"]):
        try:
            self.embedding_function = OllamaEmbeddings(model=embedding_model)
            self.chromadb_client = chromadb.HttpClient(host=host, port=port)
        except Exception as e:
            print(f"Error initializing VectorDB: {e}")

    def check_connection(self):
        return "Connected Succesfully" if self.chromadb_client.heartbeat() else "Connection Failed"
    
    def collection(self, collection_name:str):
        try:
            return self.chromadb_client.get_or_create_collection(name=collection_name)
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {e}")
            return None
    
    def remove_collection(self, collection_name: str):
        try:
            self.chromadb_client.delete_collection(name=collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")



def chunk_text(document, chunk_size:int = 500, overlap:int = 50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(document)

def input_documents(chroma_db_collection, documents:list):
    try:
        for doc in documents:
            chunks = chunk_text(doc)
            chroma_db_collection.add(
                ids=[str(uuid.uuid4()) for _ in range(len(chunks))], 
                documents=chunks,
                )
    except Exception as e:
        print(f"Error adding documents to collection: {e}")

if __name__ == "__main__":
    #Sample documents
    sample_documents = [
    """
    Artificial intelligence is changing how software systems are designed and
    operated. Traditional applications follow fixed rules written by developers,
    while AI-powered applications can interpret natural language, identify
    patterns, and generate useful responses. Modern language models can summarize
    documents, answer questions, write code, and assist with decision-making.
    However, language models do not automatically know private or recently updated
    information. Retrieval-augmented generation addresses this limitation by
    retrieving relevant information from an external knowledge source before
    producing an answer.
    """,

    """
    A vector database stores information as numerical representations called
    embeddings. An embedding captures the semantic meaning of text, allowing the
    database to compare documents based on meaning instead of exact keywords.
    For example, a search for "ways to improve application speed" may retrieve a
    document about "software performance optimization" even though the wording is
    different. During ingestion, documents are divided into smaller chunks. Each
    chunk is converted into an embedding and stored together with its original
    content and metadata.
    """,

    """
    A retrieval-augmented generation pipeline normally contains several stages.
    First, documents are loaded from sources such as PDF files, websites, databases,
    or plain text. Second, a text splitter divides the documents into manageable
    chunks with some overlap between adjacent chunks. Third, an embedding model
    converts every chunk into a vector. These vectors are stored in a vector
    database such as ChromaDB. When a user submits a question, the same embedding
    model converts the question into a vector. The database performs a similarity
    search and returns the most relevant chunks, which are then supplied to a
    language model as context for generating the final response.
    """,

    """
    Choosing an appropriate chunk size is important for retrieval quality. Very
    small chunks may lose essential context, while very large chunks may contain
    unrelated information and consume too much of the model's context window.
    Chunk overlap helps preserve information that crosses chunk boundaries. The
    correct settings depend on the document type, embedding model, and expected
    questions. Testing several chunk sizes and evaluating retrieval results is
    generally more reliable than selecting values without measurement.
    """,

    """
    ChromaDB is an open-source database designed for storing and searching
    embeddings. It supports collections, document metadata, similarity queries,
    and persistent or server-based operation. In a client-server configuration,
    an application connects to a ChromaDB server using a host and port. The
    application can add documents to a collection, retrieve similar documents,
    update existing records, and delete records. Stable identifiers should be
    assigned to stored chunks so that the source data can be updated without
    creating unwanted duplicates.
    """
    ]

    try:
        # Initialize the VectorDB instance
        vector_db = VectorDB()
        # Create a testing collection
        vector_db.remove_collection("test_collection")
        collection = vector_db.collection("test_collection")

        input_documents(collection, sample_documents)

        results = collection.query(query_texts=["what is AI"], n_results=3)

        pprint(results)
        print(len(results['documents']))

    except Exception as e:
        print(f"Error during vector database operations: {e}")

