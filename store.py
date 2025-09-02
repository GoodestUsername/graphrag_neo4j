from dataclasses import dataclass
from typing import List

import numpy as np
from neo4j import Driver
from torch import Tensor

@dataclass
class Document:
    text: List[str]
    embeddings: list[Tensor] | np.ndarray | Tensor | list[dict[str, Tensor]]


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


def store_document(driver: Driver, documents: List[Document]):
    cypher_import_query = """
        MERGE (pdf:PDF {id:$pdf_id})
        MERGE (p:Parent {id:$pdf_id + '-' + $id})
        SET p.text = $parent
        MERGE (pdf)-[:HAS_PARENT]->(p)
        WITH p, $children AS children, $embeddings as embeddings
        UNWIND range(0, size(children) - 1) AS child_index
        MERGE (c:Child {id: $pdf_id + '-' + $id + '-' + toString(child_index)})
        SET c.text = children[child_index], c.embedding = embeddings[child_index]
        MERGE (p)-[:HAS_CHILD]->(c);
    """
    driver.execute_query(
        cypher_import_query,
        id=str(i),
        pdf_id='1709.00666'
        parent=chunk,
        children=child_chunks,
        embeddings=embeddings,
    )