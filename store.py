from typing import List

import numpy as np
from neo4j import Driver
from torch import Tensor


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
