from dataclasses import dataclass
from typing import List

import numpy as np
from neo4j import Driver
from torch import Tensor

from graphrag_neo4j.embedder import Embedder


@dataclass
class Document:
    text: List[str]
    embeddings: list[Tensor] | np.ndarray | Tensor | list[dict[str, Tensor]]


@dataclass
class Section:
    id: str
    text: str


def store_node(
    driver: Driver,
    document: Document,
):
    driver.execute_query(
        """
            WITH $chunks as chunks, range(0, size($chunks)) AS index
            UNWIND index AS i
            WITH i, chunks[i] AS chunk, $embeddings[i] AS embedding
            MERGE (c:Chunk {index: i})
            SET c.text = chunk, c.embedding = embedding
        """,
        chunks=document.text,
        embeddings=document.embeddings,
    )


def store_document(
    driver: Driver, embedder: Embedder, id: str, sections: List[Section]
):
    from graphrag_neo4j.ingestion import chunk_text

    cypher_import_query = """
        MERGE (pdf:PDF {id:$pdf_id})
        MERGE (p:Parent {id:$pdf_id + '_' + $id})
        SET p.text = $parent
        MERGE (pdf)-[:HAS_PARENT]->(p)
        WITH p, $children AS children, $embeddings as embeddings
        UNWIND range(0, size(children) - 1) AS child_index
        MERGE (c:Child {id: $pdf_id + '_' + $id + '_' + toString(child_index)})
        SET c.text = children[child_index], c.embedding = embeddings[child_index]
        MERGE (p)-[:HAS_CHILD]->(c);
    """

    for section in sections:
        chunked_sections = chunk_text(section.text, 512, 64)
        embeddings = embedder.encode(chunked_sections)
        driver.execute_query(
            cypher_import_query,
            parent=section.text,
            id=section.id,
            pdf_id=id,
            children=chunked_sections,
            embeddings=embeddings,
        )
