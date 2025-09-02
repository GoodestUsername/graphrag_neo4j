from neo4j import Driver


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


def create_graph_vector_index(driver: Driver):
    driver.execute_query(
        """
            CREATE VECTOR INDEX parent IF NOT EXISTS
            FOR (c:Child)
            ON c.embedding
        """
    )
