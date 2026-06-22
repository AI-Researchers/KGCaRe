# core_utils/hybrid_context_retriever.py

import os
from typing import List, Dict, Tuple, Optional

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KGTableRetriever,
    KnowledgeGraphRAGRetriever
)

# Custom KG retrievers
from core_utils.kg_tripple_retrievers import (
    KGRetrieverToG,
    KGRetrieverToGTraversal,
    KGCaRe
)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines Vector and KG retrievers."""

    def __init__(
        self,
        vector_retriever: Optional[VectorIndexRetriever] = None,
        kg_retriever: Optional[KGTableRetriever] = None,
        reranker=None,
        mode: str = "OR",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._mode = mode
        self._reranker = reranker

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode. Choose from 'AND' or 'OR'.")

    def _retrieve(self, query_bundle: QueryBundle) -> Tuple[List[NodeWithScore], Dict[str, List[NodeWithScore]]]:
        vector_nodes, kg_nodes = [], []

        if self._vector_retriever:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            if self._reranker:
                vector_nodes = self._reranker.postprocess_nodes(vector_nodes, query_bundle)

        if self._kg_retriever:
            kg_nodes = self._kg_retriever.retrieve(query_bundle)

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        vector_ids = set(combined_dict) & {n.node.node_id for n in vector_nodes}
        kg_ids = set(combined_dict) & {n.node.node_id for n in kg_nodes}

        retrieve_ids = vector_ids & kg_ids if self._mode == "AND" else vector_ids | kg_ids
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes, {'vector': vector_nodes, 'kg': kg_nodes}


class HybridRetrieverToG(BaseRetriever):
    """Hybrid retriever using KGRetrieverToG."""

    def __init__(
        self,
        vector_retriever: Optional[VectorIndexRetriever] = None,
        kg_retriever: Optional[KGRetrieverToG] = None,
        reranker=None,
        mode: str = "OR",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._mode = mode
        self._reranker = reranker

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode. Choose from 'AND' or 'OR'.")

    def _retrieve(self, query_bundle: QueryBundle) -> Tuple[List[NodeWithScore], Dict[str, List[NodeWithScore]]]:
        vector_nodes, kg_nodes = [], []

        if self._vector_retriever:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            if self._reranker:
                vector_nodes = self._reranker.postprocess_nodes(vector_nodes, query_bundle)

        if self._kg_retriever:
            kg_nodes = self._kg_retriever._retrieve(query_bundle)

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        vector_ids = set(combined_dict) & {n.node.node_id for n in vector_nodes}
        kg_ids = set(combined_dict) & {n.node.node_id for n in kg_nodes}

        retrieve_ids = vector_ids & kg_ids if self._mode == "AND" else vector_ids | kg_ids
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes, {'vector': vector_nodes, 'kg': kg_nodes}


class HybridRetrieverKGCaRe(BaseRetriever):
    """Hybrid retriever using KGRetrieverToGTraversal_final."""

    def __init__(
        self,
        vector_retriever: Optional[VectorIndexRetriever] = None,
        kg_retriever: Optional[KGCaRe] = None,
        reranker=None,
        mode: str = "OR",
    ) -> None:
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        self._mode = mode
        self._reranker = reranker

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode. Choose from 'AND' or 'OR'.")

    def _retrieve(self, query_bundle: QueryBundle) -> Tuple[List[NodeWithScore], Dict[str, List[NodeWithScore]]]:
        vector_nodes, kg_nodes = [], []
        if self._vector_retriever:
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            if self._reranker:
                vector_nodes = self._reranker.postprocess_nodes(vector_nodes, query_bundle)
        if self._kg_retriever:
            kg_nodes = self._kg_retriever._retrieve(query_bundle)
        
        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        
        dict_retrieve_nodes = {'vector':vector_nodes, 'kg':kg_nodes}
        
        return retrieve_nodes , dict_retrieve_nodes
