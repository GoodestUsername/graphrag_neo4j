import os

from dotenv import load_dotenv
from neo4j import Driver, EagerResult, GraphDatabase

from bee_movie_script import BEE_MOVIE_SCRIPT
from db_setup import create_text_index, create_vector_index
from embedder import Embedder, EmbeddingService
from ingestion import chunk_text
from search import hybrid_search, text_search, vector_search
from store import Document, store_node

load_dotenv()


URI = os.getenv("URI", "neo4j://localhost:7687")
USER = os.getenv("USER", "neo4j")
PASSWORD = os.getenv("PASSWORD", "password")
AUTH = (USER, PASSWORD)
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "")


def store_text(driver: Driver, embedder: Embedder, text: str):
    chunks = chunk_text(text, 512, 64, False)
    store_node(driver, Document(text=chunks, embeddings=embedder.encode(chunks)))


def print_single_method_search_results(similar_results: EagerResult):
    for record in similar_results.records:
        print("text: \n" + record["text"])
        print("score: " + str(record["score"]))
        print("index: " + str(record["index"]))
        print("======")


def print_hybrid_search_results(similar_hybrid_results: EagerResult):
    for record in similar_hybrid_results.records:
        print("text: \n" + record["node"]["text"])
        print("score: " + str(record["score"]))
        print("index: " + str(record["node"]["index"]))
        print("======")


def main():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        embedder = EmbeddingService(EMBED_MODEL, cache_dir=CACHE_DIR)

        create_vector_index(driver)
        create_text_index(driver)
        store_text(driver, embedder, BEE_MOVIE_SCRIPT)

        text_search_results = text_search(driver, "according to all")
        print_single_method_search_results(text_search_results)

        vector_search_results = vector_search(driver, embedder, "according to all")
        print_single_method_search_results(vector_search_results)

        hybrid_search_results = hybrid_search(driver, embedder, "according to all")
        print_hybrid_search_results(hybrid_search_results)


if __name__ == "__main__":
    main()
