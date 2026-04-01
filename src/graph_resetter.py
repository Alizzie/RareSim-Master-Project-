"""
src/reset_graph.py

Wipes the Neo4j graph completely — all nodes, edges, constraints, and indexes.
Run this before rebuilding the knowledge graph from scratch.

Usage:
    python src/reset_graph.py
"""

import logging

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
except ImportError:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "raresim123"
    NEO4J_DATABASE = "neo4j"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def reset_graph(uri: str, user: str, password: str, database: str):
    """Connect to Neo4j and delete all nodes, relationships, constraints, and indexes."""

    log.info("Connecting to Neo4j at %s ...", uri)
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except ServiceUnavailable as e:
        raise RuntimeError(
            f"Cannot connect to Neo4j at {uri}\n"
            "Make sure the container is running:\n"
            "  docker start neo4j-raresim"
        ) from e

    with driver.session(database=database) as session:

        # Count before
        n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        n_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        log.info("Current graph: %d nodes, %d relationships", n_nodes, n_rels)

        if n_nodes == 0 and n_rels == 0:
            log.info("Graph is already empty — nothing to reset.")
            driver.close()
            return

        # Confirm
        answer = (
            input(
                f"\nThis will permanently delete {n_nodes} nodes and {n_rels} "
                f"relationships from '{database}'.\nType 'yes' to confirm: "
            )
            .strip()
            .lower()
        )

        if answer != "yes":
            log.info("Reset cancelled.")
            driver.close()
            return

        # Delete all nodes and edges
        log.info("Deleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")

        # Drop constraints
        log.info("Dropping constraints...")
        constraints = session.run("SHOW CONSTRAINTS").data()
        for c in constraints:
            name = c.get("name")
            if name:
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                log.info("  Dropped constraint: %s", name)

        # Drop indexes
        log.info("Dropping indexes...")
        indexes = session.run("SHOW INDEXES").data()
        for idx in indexes:
            name = idx.get("name")
            # Skip built-in lookup indexes Neo4j manages itself
            if name and idx.get("type") != "LOOKUP":
                session.run(f"DROP INDEX {name} IF EXISTS")
                log.info("  Dropped index: %s", name)

        # Verify
        remaining = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        log.info("Reset complete. Nodes remaining: %d", remaining)

    driver.close()


if __name__ == "__main__":
    reset_graph(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
