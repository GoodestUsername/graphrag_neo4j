import os

from dotenv import load_dotenv
from neo4j import Driver, EagerResult, GraphDatabase

from db_setup import create_graph_vector_index
from embedder import Embedder, EmbeddingService
from ingestion import split_text_to_section_by_titles
from moby_dick_text import MOBY_DICK_TEXT
from search import graph_vector_search
from store import store_document

load_dotenv()


URI = os.getenv("URI", "neo4j://localhost:7687")
USER = os.getenv("USER", "neo4j")
PASSWORD = os.getenv("PASSWORD", "password")
AUTH = (USER, PASSWORD)
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "")


def store_text(driver: Driver, embedder: Embedder, text: str, pdf_id: str):
    sections = split_text_to_section_by_titles(text)
    store_document(driver, embedder, pdf_id, sections)


def print_graph_search_results(similar_graph_results: EagerResult):
    for record in similar_graph_results.records:
        print("text: \n" + record["text"])
        print("score: " + str(record["score"]))
        print("======")


def main():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        embedder = EmbeddingService(EMBED_MODEL, cache_dir=CACHE_DIR)
        create_graph_vector_index(driver)
        store_text(driver, embedder, MOBY_DICK_TEXT, "moby_dick")
        graph_vector_search_results = graph_vector_search(
            driver, embedder, "parent", "captain", 4
        )
        print_graph_search_results(graph_vector_search_results)


if __name__ == "__main__":
    main()
