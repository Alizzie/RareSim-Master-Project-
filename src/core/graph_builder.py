"""
src/graph_builder.py

Builds the RareSim heterogeneous knowledge graph in Neo4j.
Reads from outputs/shared/ (pre-built artifacts) — does NOT re-parse raw ontology files.

Node labels:   HPOTerm, Disease
Relationships: IS_A, HAS_PHENOTYPE, SUBCLASS_OF, SAME_AS

Usage:
    python src/graph_builder.py

Requires Neo4j running locally (see README for docker command).
"""

import json
import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


try:
    from .config import (
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        NEO4J_DATABASE,
        SHARED_DIR,
    )
except ImportError:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "raresim123"
    NEO4J_DATABASE = "neo4j"
    SHARED_DIR = Path(__file__).parent.parent / "outputs" / "shared"

SHARED_DIR = Path(SHARED_DIR)
BATCH_SIZE = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_json(filename: str) -> Any:
    """Load a JSON file and return its contents."""
    file_path = SHARED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n Run build_shared_artifcats.py first."
        )

    with open(file_path, "r") as f:
        return json.load(f)


def _batched(iterable, size: int):
    """Yield successive chunks of `size` from a list."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _run_batched(session, query: str, rows: list, label: str):
    total = 0
    for batch in _batched(rows, BATCH_SIZE):
        result = session.run(query, rows=batch)
        summary = result.consume()
        total += summary.counters.nodes_created + summary.counters.relationships_created
    log.info("  %s: %d records processed", label, len(rows))
    return total


# --------------------------------------------------------------------------
# Constraints & indexes
# --------------------------------------------------------------------------


def create_constraints(session):
    log.info("Creating constraints and indexes...")
    constraints = [
        "CREATE CONSTRAINT hpo_id IF NOT EXISTS FOR (h:HPOTerm) REQUIRE h.id IS UNIQUE",
        "CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX hpo_label IF NOT EXISTS FOR (h:HPOTerm) ON (h.label)",
        "CREATE INDEX disease_label IF NOT EXISTS FOR (d:Disease) ON (d.label)",
    ]
    for stmt in constraints + indexes:
        session.run(stmt)
    log.info("  Constraints and indexes ready.")


# --------------------------------------------------------------------------
# HPO nodes
# --------------------------------------------------------------------------


def load_hpo_nodes(session, hpo_labels: dict, ic_values: dict):
    """Insert HPO term nodes with their labels and information content."""
    log.info("Inserting HPO term nodes...")

    rows = []

    for hpo_id, label in hpo_labels.items():
        rows.append(
            {
                "id": hpo_id,
                "label": label,
                "ic": ic_values.get(hpo_id, 0.0),
            }
        )

    query = """
        UNWIND $rows AS row
        MERGE (h:HPOTerm {id: row.id})
        SET h.label = row.label, h.ic = row.ic 
        """

    _run_batched(session, query, rows, "HPO nodes")


# --------------------------------------------------------------------------
# IS_A relationships between HPO terms
# --------------------------------------------------------------------------


def load_is_a_edge(session, hpo_parents: dict):
    """
    Insert IS_A relationships between HPO terms.

    hpo_parents: {child_id: [parent_id1, parent_id2, ...], ...} from outputs/shared/hpo_ancestors.json
    Creates edges like (child:HPOTerm)-[:IS_A]->(parent:HPOTerm)
    """

    log.info("Inserting IS_A relationships between HPO terms...")

    rows = []

    for child_id, parent_ids in hpo_parents.items():
        for parent_id in parent_ids:
            rows.append({"child": child_id, "parent": parent_id})

    query = """
        UNWIND $rows AS row
        MATCH (child:HPOTerm {id: row.child}), (parent:HPOTerm {id: row.parent})
        MERGE (child)-[:IS_A]->(parent)
        """

    _run_batched(session, query, rows, "IS_A edges")


# --------------------------------------------------------------------------
# Disease nodes and HAS_PHENOTYPE relationships
# --------------------------------------------------------------------------


def load_disease_nodes_and_edges(session, disease_profiles: dict):
    """
    Insert Disease nodes and HAS_PHENOTYPE relationships.

    disease_profiles: {disease_id: {"label": disease_label, "hpo_terms": [hpo_id1, hpo_id2, ...]}, ...}
    Creates Disease nodes and edges like (d:Disease)-[:HAS_PHENOTYPE]->(h:HPOTerm)
    """

    log.info("Inserting Disease nodes and HAS_PHENOTYPE relationships...")

    disease_rows = []
    edge_rows = []
    skipped_edge_rows = 0

    for disease_id, profile in disease_profiles.items():
        source_ids = profile.get("source_ids", {})

        disease_rows.append(
            {
                "id": disease_id,
                "label": profile.get("label") or profile.get("ordo_label", disease_id),
                "description": profile.get("merged_description", ""),
                "canonicalized_to_orpha": bool(
                    profile.get("canonicalized_to_orpha", False)
                ),
                "ordo_id": source_ids.get("ordo_id", ""),
                "mondo_id": source_ids.get("mondo_id", ""),
                "ordo_local_id": source_ids.get("ordo_local_id", ""),
                "mondo_local_id": source_ids.get("mondo_local_id", ""),
            }
        )

        terms = profile.get("hpo_terms", [])
        if not terms:
            skipped_edge_rows += 1
            continue

        for hpo_id in terms:
            edge_rows.append({"disease": disease_id, "hpo": hpo_id})

    # Insert Disease nodes
    disease_query = """
        UNWIND $rows AS row
        MERGE (d:Disease {id: row.id})
        SET d.label = row.label, d.description = row.description, d.canonicalized_to_orpha = row.canonicalized_to_orpha, d.ordo_id = row.ordo_id, d.mondo_id = row.mondo_id, d.ordo_local_id = row.ordo_local_id, d.mondo_local_id = row.mondo_local_id
        """
    _run_batched(session, disease_query, disease_rows, "Disease nodes")

    # Insert HAS_PHENOTYPE edges
    if skipped_edge_rows > 0:
        log.info(
            "  Skipped %d diseases with no HPO terms for HAS_PHENOTYPE edges",
            skipped_edge_rows,
        )

    edge_query = """
        UNWIND $rows AS row
        MATCH (d:Disease {id: row.disease}), (h:HPOTerm {id: row.hpo})
        MERGE (d)-[:HAS_PHENOTYPE]->(h)
        """
    _run_batched(session, edge_query, edge_rows, "HAS_PHENOTYPE edges")


# --------------------------------------------------------------------------
# SAME_AS Edges (Cross-ontology mappings)
# --------------------------------------------------------------------------


def load_same_as_edges(session, alias_to_canonical: list):
    """
    (alias_Disease)-[:SAME_AS]->(canonical_Disease)

    alias_to_canonical maps every alias ID (OMIM, MONDO, etc.) to its
    canonical ORPHA ID. We only create the edge when both nodes exist —
    aliases not loaded as Disease nodes are silently skipped.
    """

    log.info("Loading SAME_AS edges for cross-ontology mappings...")

    rows = []

    for alias_id, canonical_id in alias_to_canonical.items():
        if alias_id == canonical_id:
            continue  # Skip self-mappings

        rows.append(
            {
                "alias": alias_id,
                "canonical": canonical_id,
            }
        )

    query = """
        UNWIND $rows AS row
        MATCH (alias:Disease {id: row.alias}), (canonical:Disease {id: row.canonical})
        MERGE (alias)-[:SAME_AS]->(canonical)
        """

    _run_batched(session, query, rows, "SAME_AS edges")


# --------------------------------------------------------------------------
# Graph summary
# --------------------------------------------------------------------------


def print_graph_summary(session):
    """Print summary statistics about the graph."""
    log.info("-" * 40)
    log.info("Graph summary:")
    log.info("-" * 40)

    for label in ["HPOTerm", "Disease"]:
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
        count = result.single()["count"]
        log.info("Total %s nodes: %d", label, count)

    for rel in ["IS_A", "HAS_PHENOTYPE", "SAME_AS"]:
        result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS count")
        count = result.single()["count"]
        log.info("Total %s relationships: %d", rel, count)

    n_with_ic = session.run(
        "MATCH (h:HPOTerm) WHERE h.ic > 0 RETURN count(h) AS count"
    ).single()["count"]

    log.info("HPO terms with IC > 0: %d", n_with_ic)

    log.info("-" * 40)


# --------------------------------------------------------------------------
# Graph building workflow
# --------------------------------------------------------------------------


def build_knowledge_graph(uri: str, user: str, password: str, database: str):
    """Main function to build the knowledge graph in Neo4j."""
    log.info("Connecting to Neo4j at %s...", uri)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except ServiceUnavailable as e:
        raise ConnectionError(
            f"Failed to connect to Neo4j at {uri}. Make sure the container is running. Original error: {e}"
        ) from e

    log.info("Connected to Neo4j. Building graph...")
    log.info("Loading shared artifacts from %s...", SHARED_DIR)

    hpo_labels = load_json("hpo_labels.json")
    hpo_parents = load_json("hpo_parents.json")
    disease_profiles = load_json("canonical_disease_profiles.json")
    alias_to_canonical = load_json("alias_to_canonical.json")
    ic_values = load_json("information_content.json")

    log.info(
        "Loaded %d HPO labels, %d disease profiles, %d alias mappings",
        len(hpo_labels),
        len(disease_profiles),
        len(alias_to_canonical),
    )

    with driver.session(database=database) as session:
        create_constraints(session)
        load_hpo_nodes(session, hpo_labels, ic_values)
        load_is_a_edge(session, hpo_parents)
        load_disease_nodes_and_edges(session, disease_profiles)
        load_same_as_edges(session, alias_to_canonical)
        print_graph_summary(session)

    driver.close()
    log.info("Graph building complete.")


if __name__ == "__main__":
    build_knowledge_graph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
