import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from neo4j import Driver, EagerResult, GraphDatabase
from torch import Tensor

from embedder import Embedder, EmbeddingService
from ingestion import chunk_text
from search import hybrid_search

load_dotenv()


URI = os.getenv("URI", "neo4j://localhost:7687")
USER = os.getenv("USER", "neo4j")
PASSWORD = os.getenv("PASSWORD", "password")
AUTH = (USER, PASSWORD)
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "")


def store_node(
    driver: Driver,
    chunks: List[str],
    embeddings: list[Tensor] | np.ndarray | Tensor | list[dict[str, Tensor]],
):
    driver.execute_query(
        """
            WITH $chunks as chunks, range(0, size($chunks)) AS index
            UNWIND index AS i
            WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
            MERGE (c:Chunk {index: i})
            SET c.text = chunk, c.embedding = embedding
        """,
        chunks=chunks,
        embeddings=embeddings,
    )


def store_text(driver: Driver, embedder: Embedder, text: str):
    chunks = chunk_text(text, 512, 64, False)
    store_node(driver, chunks, embedder.encode(chunks))


def print_single_method_search_results(similar_results: EagerResult):
    for record in similar_results.records:
        print(record["text"])
        print(record["score"], record["index"])
        print("======")


def print_hybrid_search_results(similar_hybrid_results: EagerResult):
    for record in similar_hybrid_results.records:
        print("text: \n" + record["node"]["text"])
        print("score: " + record["score"])
        print("index: " + record["node"]["index"])
        print("======")


def main():
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        embedder = EmbeddingService(EMBED_MODEL, cache_dir=CACHE_DIR)
        res = hybrid_search(driver, embedder, "according to all")
        print_hybrid_search_results(res)


if __name__ == "__main__":
    main()
