import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from neo4j import Driver, EagerResult
from torch import Tensor

from embedder import Embedder

load_dotenv()

URI = os.getenv("URI", "neo4j://localhost:7687")
USER = os.getenv("USER", "neo4j")
PASSWORD = os.getenv("PASSWORD", "password")
AUTH = (USER, PASSWORD)


def chunk_text_whitespace_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    index = 0

    while len(text) > index:
        prev_whitespace = 0
        left_index = index - overlap
        while left_index >= 0:
            if text[left_index] == " ":
                prev_whitespace = left_index
                break
            left_index -= 1
        next_whitespace = text.find(" ", index + chunk_size)
        if next_whitespace == -1:
            next_whitespace = len(text)
        chunk = text[prev_whitespace:next_whitespace].strip()
        chunks.append(chunk)
        index = next_whitespace + 1
    return chunks


def chunk_text_size_split(text: str, chunk_size: int, overlap: int):
    chunks = []
    index = 0

    while len(text) > index:
        start = max(0, index - overlap + 1)
        end = min(index + chunk_size + overlap, len(text))
        chunk = text[start:end].strip()
        chunks.append(chunk)
        index += chunk_size
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int, split_on_whitespace_only=True):
    return (
        chunk_text_whitespace_split(text, chunk_size, overlap)
        if split_on_whitespace_only
        else chunk_text_size_split(text, chunk_size, overlap)
    )


def embed(text: str | List[str], embedder: Embedder):
    return embedder.encode(text)


def create_vector_index(driver: Driver):
    driver.execute_query(
        """
            CREATE VECTOR INDEX pdf IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
        """
    )


def create_text_index(driver: Driver):
    driver.execute_query(
        "CREATE FULLTEXT INDEX PdfChunkFullText FOR (c:Chunk) ON EACH [c.text]"
    )


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
    store_node(driver, chunks, embed(chunks, embedder))


def vector_search(driver: Driver, embedder: Embedder, question: str):
    return driver.execute_query(
        """
            CALL db.index.vector.queryNodes('pdf', 2, $question_embedding)
            YIELD node AS hits, score
            RETURN hits.text AS text, score, hits.index AS index
        """,
        question_embedding=embed(question, embedder=embedder)[0],
    )


def text_search(driver: Driver, question: str):
    return driver.execute_query(
        """
            CALL db.index.fulltext.queryNodes('PdfChunkFullText', $question, {limit: 2})
            YIELD node as hits, score
            RETURN hits.text AS text, score, hits.index AS index
        """,
        question=question,
    )


def hybrid_search(driver: Driver, embedder: Embedder, question: str, k: int = 2):
    question_embedding = embed(question, embedder)[0]
    return driver.execute_query(
        """
            CALL () {
                CALL db.index.vector.queryNodes('pdf', $k, $question_embedding)
                YIELD node, score
                WITH collect({node:node, score:score}) AS nodes, max(score) AS max
                UNWIND nodes AS n
                RETURN n.node AS node, (n.score / max) AS score

                UNION

                CALL db.index.fulltext.queryNodes('PdfChunkFullText', $question, {limit: $k})
                YIELD node, score
                WITH collect({node:node, score:score}) AS nodes, max(score) AS max
                UNWIND nodes AS n
                RETURN n.node AS node, (n.score / max) AS score
            }

            WITH node, max(score) AS score ORDER BY score DESC LIMIT $k
            RETURN node, score
            """,
        k=k,
        question=question,
        question_embedding=question_embedding,
    )


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
