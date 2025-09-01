from neo4j import Driver

from embedder import Embedder


def vector_search(driver: Driver, embedder: Embedder, question: str):
    return driver.execute_query(
        """
            CALL db.index.vector.queryNodes('pdf', 2, $question_embedding)
            YIELD node AS hits, score
            RETURN hits.text AS text, score, hits.index AS index
        """,
        question_embedding=embedder.encode(question)[0],
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
    question_embedding = embedder.encode(question)[0]
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
