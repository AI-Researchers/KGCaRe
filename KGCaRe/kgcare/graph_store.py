from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    GraphDatabase = None  # type: ignore[assignment]

from kgcare.config import Neo4jConfig
from kgcare.schemas import TripleRecord


class Neo4jTripleStore:
    """Neo4j store for KGCaRe triples.

    Relationship type is kept generic for Neo4j safety; the extracted relation
    text is stored on the edge and returned by traversal queries as the effective
    relation. This preserves the traversal-visible triple semantics.
    """

    def __init__(self, config: Neo4jConfig, dataset: str, index_name: str) -> None:
        self.config = config
        self.dataset = dataset
        self.index_name = index_name
        if GraphDatabase is None:
            raise ImportError("Missing dependency 'neo4j'. Install with: pip install -r KGCaRe/requirements.txt")
        self.driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))

    def close(self) -> None:
        self.driver.close()

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def ensure_constraints(self) -> None:
        self.query(
            """
            CREATE CONSTRAINT kgcare_entity_key IF NOT EXISTS
            FOR (n:KGCaReEntity)
            REQUIRE (n.dataset, n.index_name, n.id) IS UNIQUE
            """
        )

    def reset_namespace(self) -> None:
        self.query(
            """
            MATCH (n:KGCaReEntity {dataset: $dataset, index_name: $index_name})
            DETACH DELETE n
            """,
            {"dataset": self.dataset, "index_name": self.index_name},
        )

    def upsert_triples(self, triples: Iterable[TripleRecord], batch_size: int = 500) -> None:
        self.ensure_constraints()
        batch: list[dict[str, Any]] = []
        for triple in triples:
            batch.append(
                {
                    "triple_id": triple.triple_id,
                    "dataset": self.dataset,
                    "index_name": self.index_name,
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                    "doc_id": triple.doc_id,
                    "chunk_id": triple.chunk_id,
                    "source_path": triple.source_path,
                    "evidence": triple.evidence,
                    "metadata_json": json.dumps(triple.metadata, ensure_ascii=False),
                }
            )
            if len(batch) >= batch_size:
                self._upsert_batch(batch)
                batch = []
        if batch:
            self._upsert_batch(batch)

    def _upsert_batch(self, rows: list[dict[str, Any]]) -> None:
        self.query(
            """
            UNWIND $rows AS row
            MERGE (h:KGCaReEntity {dataset: row.dataset, index_name: row.index_name, id: row.head})
            SET h.name = row.head
            MERGE (t:KGCaReEntity {dataset: row.dataset, index_name: row.index_name, id: row.tail})
            SET t.name = row.tail
            MERGE (h)-[r:RELATES_TO {
                dataset: row.dataset,
                index_name: row.index_name,
                triple_id: row.triple_id
            }]->(t)
            SET r.relation = row.relation,
                r.doc_id = row.doc_id,
                r.chunk_id = row.chunk_id,
                r.source_path = row.source_path,
                r.evidence = row.evidence,
                r.metadata_json = row.metadata_json
            """,
            {"rows": rows},
        )

    def stats(self) -> dict[str, int]:
        rows = self.query(
            """
            MATCH (n:KGCaReEntity {dataset: $dataset, index_name: $index_name})
            WITH count(n) AS nodes
            MATCH (:KGCaReEntity {dataset: $dataset, index_name: $index_name})
              -[r:RELATES_TO {dataset: $dataset, index_name: $index_name}]->
              (:KGCaReEntity {dataset: $dataset, index_name: $index_name})
            RETURN nodes, count(r) AS edges
            """,
            {"dataset": self.dataset, "index_name": self.index_name},
        )
        if not rows:
            return {"nodes": 0, "edges": 0}
        return {"nodes": int(rows[0]["nodes"]), "edges": int(rows[0]["edges"])}

    def export_triples(self, limit: int = 20) -> list[tuple[str, str, str]]:
        rows = self.query(
            """
            MATCH (h:KGCaReEntity {dataset: $dataset, index_name: $index_name})
              -[r:RELATES_TO {dataset: $dataset, index_name: $index_name}]->
              (t:KGCaReEntity {dataset: $dataset, index_name: $index_name})
            RETURN h.id AS head_entity, r.relation AS relation, t.id AS tail_entity
            LIMIT $limit
            """,
            {"dataset": self.dataset, "index_name": self.index_name, "limit": limit},
        )
        return [(row["head_entity"], row["relation"], row["tail_entity"]) for row in rows]
