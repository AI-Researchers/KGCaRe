from typing import Optional, Callable, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from typing import List, Dict
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings

from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
DEFAULT_NODE_SCORE = 1000.0
GLOBAL_EXPLORE_NODE_LIMIT = 3
REL_TEXT_LIMIT = 30

class KGRetrieverToG(BaseRetriever):
    def __init__(
        self, 
        index : KnowledgeGraphIndex,
        graph_store: GraphStore, 
        llm: LLM,
        embed_model: Optional[BaseEmbedding] = None,
        max_depth: int = 3,
        max_entities: int = 5,
        max_keywords_per_query: int = 10
    ):
        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore
        
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self.graph_store = graph_store
        self.llm = llm
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.max_keywords_per_query = max_keywords_per_query
        self.query_keyword_extract_template =  DQKET

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        # prompt_template = "Extract relevant keywords from the question below.\nQuestion: {question}\nKEYWORDS:"
        response = self.llm.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _entity_relation_search(self, entity: str) -> List[List[str]]:
        """Retrieve relationships for a given entity."""
        try:
            relations = self.graph_store.get(entity)
            # if not relations:
                # print(f"No relations found for entity: {entity}")
            return relations
        except Exception as e:
            # print(f"Error during relation search for entity '{entity}': {e}")
            return []

    def _prune_relations(self, entity: str, relations: list, question: str):
        prompt = (
            f"You are an expert in knowledge graph pruning. Based on the given question, identify and retain only the most relevant relations. Triples are in the format (Entity1, Relation, Entity2)\n"
            f"Entity1: '{entity}'\n"
            f"Question: '{question}'\n"
            f"Relations , Entity2 : {relations}\n"
            f"Output format: Provide a list of pruned relations, one per line, that are most relevant to the question."
        )
        response = self.llm.complete(prompt)
        # print("Pruned relations:", response)
        return response.text.splitlines() 

    def _reasoning(self, question: str, path: list):
        prompt = (
            f"You are an advanced reasoning system. Determine if the provided knowledge path is sufficient to answer the question.\n"
            f"Given Question: '{question}'\n"
            f"Knowledge Path: {path}\n"
            f"Is the knowledge path sufficient to answer the question accurately? Provide 'Yes' or 'No' as the response."
        )
        response = self.llm.complete(prompt).text.strip().lower()
        if "yes" in response:
            return "Yes"
        elif "no" in response:
            return "No"
        else:
            raise ValueError("Unexpected response from reasoning step. Ensure the LLM is configured correctly.")

    def _retrieve(self, query_bundle: QueryBundle) -> Dict[str, List[tuple]]:
        """Retrieve all possible knowledge paths for the given query."""
        question = query_bundle.query_str
        topic_entities = self._get_keywords(question)

        if not isinstance(topic_entities, list) or not topic_entities:
            raise ValueError("Failed to extract topic entities. Ensure the query contains relevant information.")

        depth = 0
        candidate_entities = topic_entities
        all_knowledge_paths = {}

        while depth < self.max_depth:
            knowledge_path = []
            new_candidates = []
            for entity in candidate_entities:
                relations_entity = self._entity_relation_search(entity)
                pruned_relations = self._prune_relations(entity, relations_entity, question)

                for relation in pruned_relations:
                    # Ensure that the relation is a tuple or list with at least two elements
                    if isinstance(relation, (tuple, list)) and len(relation) >= 2:
                        targets = [t[1] for t in pruned_relations if isinstance(t, (tuple, list)) and len(t) >= 2 and t[0] == relation[0]]
                        new_candidates.extend(targets)
                        knowledge_path.append((entity, relation[0], targets))

            all_knowledge_paths[f"Depth {depth+1}"] = knowledge_path

            if self._reasoning(question, knowledge_path) in ["Yes"]:
                break # Exit loop if reasoning indicates the path is sufficient
            candidate_entities = new_candidates[:self.max_entities]
            depth += 1
            
        # Final LLM-based filtering for relevant paths and triples
        prompt = (
            f"Given the question: \n '{question}', \n and the following knowledge paths:\n"
            f"{all_knowledge_paths}\n"
            f"Identify the triples and knowledge paths that are sufficient to correctly answer the question.\n"
            f"Output format:\n"
            f"Triples:\n"
            f"- List each triple on a new line in the format (Entity1, Relation, Entity2).\n"
        )
        response = self.llm.complete(prompt)
        rel_text_node = f"KG context:\n{response.text}\n"
        
        return [
                    NodeWithScore(
                        node=TextNode(text=rel_text_node), score=DEFAULT_NODE_SCORE
                    )
                ]


from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import List, Dict, Optional, Set
import re

DEFAULT_NODE_SCORE = 1000.0

class KGRetrieverToGTraversal(BaseRetriever):
    def __init__(
        self,
        index : KnowledgeGraphIndex,
        graph_store: GraphStore,
        llm: LLM,
        embed_model: Optional[BaseEmbedding] = None,
        max_depth: int = 3,
        max_entities: int = 20,
    ):
        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore
        self.graph_store = graph_store
        self.llm = llm
        self._embed_model = embed_model or Settings.embed_model
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.max_keywords_per_query = 20
        self.query_keyword_extract_template =  DQKET

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        topic_entities = self._get_keywords(query_str)
        
        if not topic_entities:
            raise ValueError("No topic entities found in query.")

        visited_entities: Set[str] = set()
        knowledge_paths = {}
        depth = 1

        while depth <= self.max_depth and topic_entities:
            all_entity_triples = set()
            new_entities = set()
            
            for entity in topic_entities:
                entity = entity.replace("-", " ")
                
                if depth == 1:
                    retrieved_triples = self._search_kg_partial(entity, visited_entities, all_entity_triples)
                else:
                    retrieved_triples = self._search_kg_exact(entity, visited_entities, all_entity_triples)
                
                visited_entities.add(entity.lower())
                all_entity_triples.update(retrieved_triples)
        
            if len(all_entity_triples) > 100:
                all_entity_triples = list(all_entity_triples)[:30]
                all_entity_triples = set(all_entity_triples)
                
            pruned_triples = self._prune_triples(query_str, all_entity_triples)
            knowledge_paths[f"Depth {depth}"] = pruned_triples

            new_entities = self._get_next_candidate_entities(pruned_triples, visited_entities)
            topic_entities = list(new_entities)[:self.max_entities]
            depth += 1

            if self._reasoning(query_str, pruned_triples):
                break

        rel_text_node = f"KG context:\n{knowledge_paths}\n"
        
        
        # Final LLM-based filtering for relevant paths and triples
        prompt = (
            f"Given the question:\n'{query_str}'\n\nand the following knowledge paths:\n{knowledge_paths}\n\nExtract and list all the triples from these knowledge paths that are necessary to fully answer the question. Ensure that every triple needed is included.\n\nOutput format:\nTriples:\n- Write each triple on a new line in the format (Entity1, Relation, Entity2) with no additional commentary."
        )
        response = self.llm.complete(prompt)
        rel_text_node = f"KG context:\n{response.text}\n"
        
        return [
                    NodeWithScore(
                        node=TextNode(text=rel_text_node), score=DEFAULT_NODE_SCORE
                    )
                ]

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        # prompt_template = "Extract relevant keywords from the question below.\nQuestion: {question}\nKEYWORDS:"
        response = self.llm.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _search_kg_partial(self, entity: str, visited_entities: Set[str], total_triples: Set[tuple]) -> Set[tuple]:
        regex_pattern = r"(?i).*(" + "|".join([f"{word.strip()}" for word in entity.split()]) + r").*"
        cypher_query = (
            "MATCH (headEntity)-[relation]->(entity) "
            "WHERE toLower(entity.id) =~ $re "  # Corrected 'name' to 'id' property
            "RETURN headEntity.id AS head_entity, type(relation) AS relation, entity.id AS tail_entity"
        )
        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            return {(r['head_entity'], r['relation'], r['tail_entity']) for r in results}
        except Exception as e:
            print(f"Error during partial search: {e}")
            return set()


    def _search_kg_exact(self, entity: str, visited_entities: Set[str], total_triples: Set[tuple]) -> Set[tuple]:
        regex_pattern = r"(?i)\\b(" + "|".join([re.escape(word.strip()) for word in entity.split()]) + r")\\b"
        cypher_query = (
            "MATCH (entity)-[relation]->(tailEntity) "
            "WHERE toLower(entity.id) =~ $re "  # Corrected 'name' to 'id' property
            "RETURN entity.id AS head_entity, type(relation) AS relation, tailEntity.id AS tail_entity"
        )
        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            return {(r['head_entity'], r['relation'], r['tail_entity']) for r in results}
        except Exception as e:
            print(f"Error during exact search: {e}")
            return set()

    def _prune_triples(self, query_str: str, triples: Set[tuple]) -> List[tuple]:
        prompt = (
            f"You are an expert in knowledge graph pruning. Based on the given question, identify and retain only the most relevant relations. Triples are in the format (Entity1, Relation, Entity2)\n"
            f"Question: '{query_str}'\n"
            f"Triples: {list(triples)}\n"
            f"Prune irrelevant triples and return only those relevant to answering the question."
            f"Output format: Provide a list of pruned triples, one per line, that are most relevant to the question. no other information /explanation  is needed."
        )
        response = self.llm.complete(prompt).text.strip()
        return [tuple(triple.split(',')) for triple in response.split('\n') if triple] if response else []

    def _get_next_candidate_entities(self, triples: List[tuple], visited_entities: Set[str]) -> Set[str]:
        candidates = set()
        for triple in triples:
            if len(triple) == 3:
                head, relation, tail = triple
                if head not in visited_entities:
                    candidates.add(head)
                if tail not in visited_entities:
                    candidates.add(tail)
        return candidates

    def _reasoning(self, query_str: str, triples: List[tuple]) -> bool:
        if not triples:
            return False

        prompt = (
            f"Does the following knowledge path sufficiently answer the question?\n"
            f"Question: '{query_str}'\n"
            f"Triples: {triples}\n"
            f"Answer with 'Yes' or 'No':"
        )
        try:
            response = self.llm.complete(prompt).text.strip().lower()
            return "yes" in response
        except Exception as e:
            print(f"Error during reasoning: {e}")
            return False
        
        
        

DEFAULT_NODE_SCORE = 1000.0

class KGRetrieverToGTraversal_v2(BaseRetriever):
    def __init__(
        self,
        index: KnowledgeGraphIndex,
        graph_store: GraphStore,
        llm: LLM,
        embed_model: Optional[BaseEmbedding] = None,
        max_depth: int = 3,
        max_entities: int = 20,
    ):
        # Ensure the index is a KnowledgeGraphIndex.
        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore
        self.graph_store = graph_store
        self.llm = llm
        # Use provided embedding model or default one.
        self._embed_model = embed_model or Settings.embed_model  
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.max_keywords_per_query = 20
        # Template for extracting keywords from query.
        self.query_keyword_extract_template = DQKET  

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # --- Initialization Step ---
        query_str = query_bundle.query_str
        # Extract topic entities from the query using LLM.
        topic_entities = self._get_keywords(query_str)
        if not topic_entities:
            raise ValueError("No topic entities found in query.")

        # Initialize visited entities to avoid revisiting nodes.
        visited_entities: Set[str] = set()
        # Memory to store all triples found over iterations.
        overall_memory: List[tuple] = []
        # Memory to store traversal paths per depth.
        knowledge_paths: Dict[str, List[tuple]] = {}
        # Initialize clue entities and clue memory as empty.
        clue_entities: List[str] = []
        clue_memory: Dict[str, List[tuple]] = {}

        depth = 1  # Initialize current traversal depth.

        # --- Iterative Traversal Loop ---
        while depth <= self.max_depth and (topic_entities or clue_entities):
            # Set to store all triples discovered in this iteration.
            all_entity_triples = set()

            # Process topic entities:
            for entity in topic_entities:
                # Clean up the entity string.
                entity = entity.replace("-", " ")
                if depth == 1:
                    # At depth 1, use a partial match search on the KG.
                    retrieved_triples = self._search_kg_partial(entity, visited_entities, all_entity_triples)
                else:
                    # At deeper levels, use an exact match search.
                    retrieved_triples = self._search_kg_exact(entity, visited_entities, all_entity_triples)
                # Mark the topic entity as visited.
                visited_entities.add(entity.lower())
                # Add any retrieved triples to the current iteration’s set.
                all_entity_triples.update(retrieved_triples)

            # Process clue entities:
            for entity in clue_entities:
                # For clue entities, always perform a partial search that checks both
                # entity and relation positions in the KG.
                retrieved_triples = self._search_kg_clue(entity, visited_entities, all_entity_triples)
                visited_entities.add(entity.lower())
                all_entity_triples.update(retrieved_triples)

            # Update overall memory with new triples found in this iteration.
            overall_memory.extend(list(all_entity_triples))

            # --- Triple Pruning Step ---
            # Use LLM to prune irrelevant triples based on the query.
            pruned_triples = self._prune_triples(query_str, all_entity_triples)

            # Update the path traversal memory with the pruned triples for this depth.
            knowledge_paths[f"Depth {depth}"] = pruned_triples

            # --- Candidate Entity Update Step ---
            # Extract candidate entities from the pruned triples for next round.
            candidate_entities = self._get_next_candidate_entities(pruned_triples, visited_entities)
            # Update topic entities (limit to max_entities).
            topic_entities = list(candidate_entities)[:self.max_entities]

            # --- Reasoning Step ---
            if pruned_triples:
                # Convert clue memory to a string (if available) to supply extra context.
                clue_memory_str = f"Clue Memory: {clue_memory}" if clue_memory else ""
                # Ask LLM if the current pruned triples (knowledge path) sufficiently answer the query.
                if self._reasoning(query_str, pruned_triples, clue_memory_str):
                    # If sufficient, break out of the loop.
                    break
                else:
                    # If not sufficient, update clue entities and clue memory.
                    new_clue_entities = self._update_clue_entities(pruned_triples, visited_entities)
                    clue_entities = list(new_clue_entities)[:self.max_entities]
                    # Save the current pruned triples in the clue memory.
                    clue_memory[f"Depth {depth}"] = pruned_triples
            else:
                # If no triples were pruned, perform reasoning with only the query.
                if self._reasoning(query_str, pruned_triples, ""):
                    break
                else:
                    # Update clue entities and memory from the overall memory.
                    new_clue_entities = self._update_clue_entities(overall_memory, visited_entities)
                    clue_entities = list(new_clue_entities)[:self.max_entities]
                    clue_memory[f"Depth {depth}"] = []  # No new clues found at this depth.
                    # If neither topic nor clue entities are available, exit the loop.
                    if not topic_entities and not clue_entities:
                        break

            depth += 1  # Increment depth for next iteration.

        # --- Final Answer Generation Step ---
        # Use LLM to generate the final answer based on all gathered knowledge paths.
        final_prompt = (
            f"Given the question:\n'{query_str}'\n\nand the following knowledge paths:\n{knowledge_paths}\n\n"
            "Extract and list all the triples from these knowledge paths that are necessary to fully answer the question. "
            "Ensure that every triple needed is included.\n\n"
            "Output format:\nTriples:\n- Write each triple on a new line in the format (Entity1, Relation, Entity2) with no additional commentary."
        )
        final_response = self.llm.complete(final_prompt)
        final_text = f"KG context:\n{final_response.text}\n"

        # Return the final answer wrapped in a NodeWithScore.
        return [
            NodeWithScore(
                node=TextNode(text=final_text), 
                score=DEFAULT_NODE_SCORE
            )
        ]

    def _get_keywords(self, query_str: str) -> List[str]:
        # Use LLM to extract keywords from the question.
        response = self.llm.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

    def _search_kg_partial(self, entity: str, visited_entities: Set[str], total_triples: Set[tuple]) -> Set[tuple]:
        # Create a regex pattern for partial matching based on words in the entity.
        regex_pattern = r"(?i).*(" + "|".join([f"{word.strip()}" for word in entity.split()]) + r").*"
        # Cypher query to match nodes partially (using the entity in the tail position).
        cypher_query = (
            "MATCH (headEntity)-[relation]->(entity) "
            "WHERE toLower(entity.id) =~ $re "  # Using the id property.
            "RETURN headEntity.id AS head_entity, type(relation) AS relation, entity.id AS tail_entity"
        )
        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            return {(r['head_entity'], r['relation'], r['tail_entity']) for r in results}
        except Exception as e:
            print(f"Error during partial search: {e}")
            return set()

    def _search_kg_exact(self, entity: str, visited_entities: Set[str], total_triples: Set[tuple]) -> Set[tuple]:
        # Create a regex pattern for exact matching (word boundary based).
        regex_pattern = r"(?i)\b(" + "|".join([re.escape(word.strip()) for word in entity.split()]) + r")\b"
        # Cypher query to match nodes exactly (using the entity in the head position).
        cypher_query = (
            "MATCH (entity)-[relation]->(tailEntity) "
            "WHERE toLower(entity.id) =~ $re "  # Using the id property.
            "RETURN entity.id AS head_entity, type(relation) AS relation, tailEntity.id AS tail_entity"
        )
        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            return {(r['head_entity'], r['relation'], r['tail_entity']) for r in results}
        except Exception as e:
            print(f"Error during exact search: {e}")
            return set()

    def _search_kg_clue(self, entity: str, visited_entities: Set[str], total_triples: Set[tuple]) -> Set[tuple]:
        # Build a regex pattern to search for the clue entity in either head, tail, or relation.
        regex_pattern = r"(?i).*(" + "|".join([re.escape(word.strip()) for word in entity.split()]) + r").*"
        # Cypher query to search in head entity, tail entity, or relation type.
        cypher_query = (
            "MATCH (headEntity)-[relation]->(entity) "
            "WHERE toLower(headEntity.id) =~ $re OR toLower(entity.id) =~ $re OR toLower(type(relation)) =~ $re "
            "RETURN headEntity.id AS head_entity, type(relation) AS relation, entity.id AS tail_entity"
        )
        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            return {(r['head_entity'], r['relation'], r['tail_entity']) for r in results}
        except Exception as e:
            print(f"Error during clue search: {e}")
            return set()

    def _prune_triples(self, query_str: str, triples: Set[tuple]) -> List[tuple]:
        # Use LLM to prune irrelevant triples given the query and the current set of triples.
        prompt = (
            f"You are an expert in knowledge graph pruning. Based on the given question, identify and retain only the most relevant relations. "
            f"Triples are in the format (Entity1, Relation, Entity2).\n"
            f"Question: '{query_str}'\n"
            f"Triples: {list(triples)}\n"
            "Prune irrelevant triples and return only those relevant to answering the question.\n"
            "Output format: Provide a list of pruned triples, one per line, that are most relevant to the question with no additional explanation."
        )
        response = self.llm.complete(prompt).text.strip()
        # Split the response into lines and convert each to a tuple.
        return [tuple(triple.split(',')) for triple in response.split('\n') if triple] if response else []

    def _get_next_candidate_entities(self, triples: List[tuple], visited_entities: Set[str]) -> Set[str]:
        # Extract candidate entities from each triple that have not been visited.
        candidates = set()
        for triple in triples:
            if len(triple) == 3:
                head, relation, tail = triple
                if head.lower() not in visited_entities:
                    candidates.add(head)
                if tail.lower() not in visited_entities:
                    candidates.add(tail)
        return candidates

    def _update_clue_entities(self, triples: List[tuple], visited_entities: Set[str]) -> Set[str]:
        # Similar to _get_next_candidate_entities, extract new entities from the pruned triples for clues.
        new_clues = set()
        for triple in triples:
            if len(triple) == 3:
                head, relation, tail = triple
                if head.lower() not in visited_entities:
                    new_clues.add(head)
                if tail.lower() not in visited_entities:
                    new_clues.add(tail)
        return new_clues

    def _reasoning(self, query_str: str, triples: List[tuple], clue_memory_str: str = "") -> bool:
        # Use LLM to decide if the current knowledge (triples) along with clue memory is sufficient to answer the question.
        if not triples:
            return False
        prompt = (
            f"Does the following knowledge path sufficiently answer the question?\n"
            f"Question: '{query_str}'\n"
            f"Triples: {triples}\n"
            f"{clue_memory_str}\n"
            "Answer with 'Yes' or 'No':"
        )
        try:
            response = self.llm.complete(prompt).text.strip().lower()
            return "yes" in response
        except Exception as e:
            print(f"Error during reasoning: {e}")
            return False



from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)

from typing import List, Dict, Optional, Set, Tuple
import re
from nltk.corpus import stopwords
import string
import ast

DEFAULT_NODE_SCORE = 1000.0

class KGRetrieverToGTraversal_final(BaseRetriever):
    
    
    # Partial match queries
    partial_match_search_head_entities = """
    MATCH (headEntity)-[relation]->(entity)
    WHERE toLower(entity.id) =~ $re
    RETURN headEntity.id AS head_entity, type(relation) AS relation, entity.id AS tail_entity
    """

    partial_match_search_tail_entities = """
    MATCH (entity)-[relation]->(tailEntity)
    WHERE toLower(entity.id) =~ $re
    RETURN entity.id AS head_entity, type(relation) AS relation, tailEntity.id AS tail_entity
    """

    partial_match_search_rel_entities = """
    MATCH (headEntity)-[relation]->(tailEntity)
    WHERE toLower(type(relation)) =~ $re
    RETURN headEntity.id AS head_entity, type(relation) AS relation, tailEntity.id AS tail_entity
    """

    # Exact match queries
    exact_match_search_head_entities = """
    MATCH (headEntity)-[relation]->(entity)
    WHERE toLower(entity.id) =~ $re
    RETURN headEntity.id AS head_entity, type(relation) AS relation, entity.id AS tail_entity
    """

    exact_match_search_tail_entities = """
    MATCH (entity)-[relation]->(tailEntity)
    WHERE toLower(entity.id) =~ $re
    RETURN entity.id AS head_entity, type(relation) AS relation, tailEntity.id AS tail_entity
    """

    # prompts 

    prune_triple_prompt = """Perform following tasks:
    1. Carefully review the question below. 
    2. From the list of available triples, select triples that you believe are most likely to help answer the provided question. 
    3. For each selected triple, provide a score between 0 to 10 reflecting its usefulness in answering the question, with 10 being most useful. 
    4. Provide a brief explanation for your choices, highlighting how each selected triple potentially contributes to answering the question.
    5. Start each answer with count of triples with score greater than or equal to 8 

    Below is an example:
    Question: Before my father died last year  June 2021, he appointed as his family property executor. I have inherited all our family properties. When can i start paying the inheritance tax ?

    Knowledge triplets:
    (Personal_representative, MUST_FILL_IN, Form_cfo_iht1_to_apply_for_inheritance_tax_payments)
    (Personal_representative, CAN_APPLY_FOR, Inheritance_tax_payments_from_the_account)
    (Inheritance_tax, MAY_BE_APPLICABLE_IF, 'the_deceased_person’s_estate_can’t_or_doesn’t_pay')
    (Heir, MAY_HAVE_TO_PAY, 'inheritance_tax')
    (Estate, CAN’T_OR_DOESN’T_PAY, 'inheritance_tax')
    (Inheritance Tax, paid by, end of sixth month)
    (Executor_of_the_will, RESPONSIBLE_FOR, Paying_inheritance_tax_from_estate)
    (Executor_of_the_will, SHOULD_PAY, Inheritance_tax_out_of_the_estate)
    (If_inheritor, SELLS_PROPERTY, Then_must_notify_hmrc_about_main_home)
    (If_inheritor, FAILS_TO_NOTIFY_HMRC, Then_hmrc_will_decide_main_home)
    (Deceased, OWNED, Property)
    (You, MAY_HAVE_TO_TELL, Land_registry_about_death_of_property_owner)
    (You, MAY_HAVE_TO_SELL, Shares_or_property_to_pay_tax_and_debts)

    Answer:
    1. (Personal_representative, MUST_FILL_IN, Form_cfo_iht1_to_apply_for_inheritance_tax_payments) (Score: 8) - Explanation: This triple is useful as it specifies a form that the personal representative must complete to apply for inheritance tax payments. This provides procedural information directly related to handling the inheritance tax.
    2. (Inheritance Tax, paid by, end of sixth month) (Score: 10) - Explanation: This triple is crucial as it gives a clear deadline for when the inheritance tax must be paid, directly addressing the timing aspect of the user's question about when they need to start paying the inheritance tax.
    3. (Executor_of_the_will, RESPONSIBLE_FOR, Paying_inheritance_tax_from_estate) (Score: 9) -  Explanation: This triple is highly relevant as it clarifies the executor's role in paying the inheritance tax from the estate, which is essential for understanding who is responsible for the tax payments.
    4. (Heir, MAY_HAVE_TO_PAY, 'inheritance_tax') (Score: 7) - Explanation: This triple is moderately useful as it indicates that the heir might be required to pay the inheritance tax. While it doesn't specify when, it sets the expectation that payment could be required.
    5. (You, MAY_HAVE_TO_SELL, Shares_or_property_to_pay_tax_and_debts) (Score: 6) - Explanation: This triple adds value by indicating a potential action (selling shares or property) that may be necessary to fulfill tax obligations, providing practical insight into managing tax payment.
    """

    reasoning_prompt = """Perform following tasks:
    1. Given a question, some clues and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to evaluate if using only these resources, are sufficient to formulate an answer ({Yes} or {No}). 
    2. Your answer must begin with {Yes} or {No}.
    3. If {Yes}, please note that the analyzed answer entity must be enclosed in curly brackets {xxxxxx}
    4. If {No}, this means the resources are insufficient or provide clues that are helpful but inconclusive for answering the question. Predict additional evidence that needs to be found to answer the current question and enclose these entities in curly brackets {xxxxxx}. 
    5. Your answer MUST NOT be the same as the one already provided in the Clue section of the question. 
    6. Treat the current clues and evidence as complete and final. 
    7. If no further unique clues can be generated based on the current information, explicitly state: {No additional unique clues can be provided}.
    8. You MUST ONLY use the given information and not your own knowledge to judge if we can answer these questions.

    Here are some examples:
    # Example 1:
    Question:
    I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category. What age can I apply for a certificate?
    Clues:
    To answer this question, evidence is needed regarding the {age requirements} and {eligibility criteria} for applying for a {gender recognition certificate} and any specific conditions that apply to individuals who are {under 18}.
    Knowledge triplets:
    Candidate, apply by, standard route
    Candidate, age requirement, 18 or over
    # Answer:
    {No} The current information does not provide details about eligibility criteria or specific conditions for individuals under 18. Additional evidence needed: {legal exceptions} or {special provisions} for obtaining gender recognition certificate for {minors}.

    Now, please carefully consider the following case:
    Question: 
    """

    reasoning_without_kg_prompt = """Given a question, identify specific, concise entities essential for answering the question and enclose each in curly brackets {xxxx}. Keep entities as short as possible, focusing on core concepts.
    Here are some examples:
    # Example 1:
    Question:
    Before my father died last year  June 2021, he appointed as his family property executor. I have inherited all our family properties. When can i start paying the inheritance tax ?
    # Answer:
    To determine when to start paying inheritance tax, it’s essential to know the {inheritance date} and the {jurisdiction} where the tax laws apply. Additionally, {tax regulations}, including any {grace periods} or {deadlines}, are important, along with {asset transfer process} details and whether any {legal consultation} was provided for this inheritance.

    # Example 2:
    Question:
    I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category. What age can I apply for a certificate?
    # Answer:
    To answer this question, evidence is needed regarding the {age requirements} and {eligibility criteria} for applying for a {gender recognition certificate} and any specific conditions that apply to individuals who are {under 18}.

    Now, please carefully consider the following case:
    Question: 
    """
    
    
    def __init__(
        self,
        index: KnowledgeGraphIndex,
        graph_store: GraphStore,
        llm: LLM,
        embed_model: Optional[BaseEmbedding] = None,
        max_depth: int = 3,
        max_entities: int = 20,
    ):
        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore
        self.graph_store = graph_store
        self.llm = llm
        self._embed_model = embed_model or Settings.embed_model
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.max_keywords_per_query = 20
        self.query_keyword_extract_template = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        topic_entities = self._get_keywords(query_str)
        if not topic_entities:
            raise ValueError("No topic entities found in query.")

        visited_entities: Set[str] = set()
        knowledge_paths = {}
        clue_memory = ""
        path_traversal_memory = ""
        memory = ""
        clue_entities = []
        intermediate_answer = {}
        new_path_dict = {}
        depth = 1

        while depth <= self.max_depth and (topic_entities or clue_entities):
            print(f"\n=== DEPTH: {depth} ===")
            all_entity_triples = set()
            all_ent_results_dict = {"matched_entities": {}}
            candidate_entities = set()
            path_traversed = []

            # --- Process topic entities ---
            for entity in topic_entities:
                entity = entity.replace("-", " ")
                if depth == 1:
                    retrieved_triples, retrieved_ent_results = self._search_kg_partial(entity, visited_entities, all_entity_triples)
                else:
                    retrieved_triples, retrieved_ent_results = self._search_kg_exact(entity, visited_entities, all_entity_triples)

                all_entity_triples, visited_entities = self._update_visited_entities_and_triples(
                    retrieved_triples, entity, retrieved_ent_results, all_entity_triples, visited_entities, all_ent_results_dict
                )

            # --- Process clue entities ---
            for entity in clue_entities:
                entity = entity.replace("-", " ")
                retrieved_triples, retrieved_ent_results = self._search_kg_partial(entity, visited_entities, all_entity_triples)

                all_entity_triples, visited_entities = self._update_visited_entities_and_triples(
                    retrieved_triples, entity, retrieved_ent_results, all_entity_triples, visited_entities, all_ent_results_dict
                )

            # --- Triple pruning ---
            pruned_triples = self._prune_triples(query_str, all_entity_triples)
            # print("triples after pruning = ", len(pruned_triples), pruned_triples)

            knowledge_paths[f"Depth {depth}"] = pruned_triples

            # --- Candidate entity + path updates ---
            candidate_entities, path_traversed, new_path_dict, pruned_triple_list = self._save_candidates_for_next_round(
                pruned_triples, all_ent_results_dict, new_path_dict, depth, memory, path_traversal_memory
            )
            
            # print("\n-- SAVE CANDIDATES OUTPUT --")
            # print(f"Candidate Entities ({len(candidate_entities)}): {candidate_entities}")
            # print(f"Path Traversed ({len(path_traversed)}):")
            # for path in path_traversed:
            #     print(f"  {path}")
            # print(f"New Path Dict (keys): {list(new_path_dict.keys())}")
            # print(f"Pruned Triple List ({len(pruned_triple_list)}): {pruned_triple_list}")


            topic_entities = list(set(candidate_entities - visited_entities))[:50]           #[:self.max_entities]

            memory = self._update_memory(memory, pruned_triple_list)
            path_traversal_memory += "\n" + "\n".join(path_traversed)
            # print(" MEMORY:\n", memory)
            # print("PATH TRAVERSAL MEMORY:\n", path_traversal_memory)

            current_depth_triple = f"Depth_{depth}_triples"
            intermediate_answer[current_depth_triple] = pruned_triple_list

            # --- Reasoning ---
            if pruned_triples:
                stop, results = self._reasoning(query_str, pruned_triple_list, clue_memory)
                # print("REASONING RESULT: ", stop, results)

                if stop:
                    # print(f"ToG stopped at depth {depth} due to reasoning.")
                    final_answer = self._extract_answer(results)
                    final_text = f"Answer: {final_answer}\n\nKG context:\n{memory}\n"
                    return [NodeWithScore(node=TextNode(text=final_text), score=DEFAULT_NODE_SCORE)]
                    break

                clues, possible_entities = self._clean_reasoning_output(results)
                current_depth_clue = f"Depth_{depth}_clues"
                intermediate_answer[current_depth_clue] = clues

                clue_memory += "\n" + clues
                clue_entities = list(set(possible_entities) - visited_entities)

            else:
                results = self._reasoning_without_kg(query_str)
                # print("REASONING WITHOUT KG: ", results)

                clue_memory += "\n" + results
                possible_entities = self._clean_reasoning_without_kg_output(results)
                clue_entities = list(set(possible_entities) - visited_entities)

                if not clue_entities and not topic_entities:
                    break

            # --- Update visited entities ---
            visited_entities.update(topic_entities)
            visited_entities.update(clue_entities)

            depth += 1

        final_text = f"KG context:\n{memory}\n"

        return [NodeWithScore(node=TextNode(text=final_text), score=DEFAULT_NODE_SCORE)]
    
    def _get_keywords(self, query_str: str) -> List[str]:
        # Use LLM to extract keywords from the question.
        response = self.llm.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(
            response, start_token="KEYWORDS:", lowercase=False
        )
        return list(keywords)

        # --- Knowledge Graph Search ---
        
    def _search_kg_partial(
        self,
        entity: str,
        visited_entities: Set[str],
        total_triples: Set[Tuple[str, str, str]]
    ) -> Tuple[Set[Tuple[str, str, str]], Dict[str, Any]]:
        # print("Searching KG partial--")
        total_results_dict = {"matched_entities": {}}

        entity_words = entity.split()
        new_entity_words = [
            new_word.replace('.', '').lower()
            for item in entity_words
            if (new_word := self._remove_stopwords(item))
        ]
        entity_words_not_visited = list(set(new_entity_words) - visited_entities)

        if len(entity_words_not_visited) > 0:
            head_match_triples, total_triples_found = self.execute_search_kg_partial(
                self.partial_match_search_head_entities,
                entity_words_not_visited,
                entity,
                "search_for_heads",
                total_triples
            )

            tail_match_triples, total_triples_found = self.execute_search_kg_partial(
                self.partial_match_search_tail_entities,
                entity_words_not_visited,
                entity,
                "search_for_tails",
                total_triples
            )

            rel_match_triples, total_triples_found = self.execute_search_kg_partial(
                self.partial_match_search_rel_entities,
                entity_words_not_visited,
                entity,
                "search_subj_obj",
                total_triples
            )

            total_results_dict["matched_entities"] = (
                head_match_triples["matched_entities"]
                | tail_match_triples["matched_entities"]
                | rel_match_triples["matched_entities"]
            )

            # print("len of total triples after partial matching = ", len(total_triples_found))
            return total_triples, total_results_dict

        else:
            return total_triples, total_results_dict



    def _search_kg_exact(
        self,
        entity: str,
        visited_entities: Set[str],
        total_triples: Set[Tuple[str, str, str]]
    ) -> Tuple[Set[Tuple[str, str, str]], Dict[str, Any]]:
        # print("Searching KG exact--")
        total_results_dict = {"matched_entities": {}}

        entity_words = entity.split()
        new_entity_words = [
            new_word.replace('.', '').lower()
            for item in entity_words
            if (new_word := self._remove_stopwords(item))
        ]
        entity_words_not_visited = list(set(new_entity_words) - visited_entities)

        if len(entity_words_not_visited) > 0:
            head_match_triples, total_triples_found = self.execute_search_kg_exact(
                self.exact_match_search_head_entities,
                entity_words_not_visited,
                entity,
                "search_for_heads",
                total_triples
            )

            tail_match_triples, total_triples_found = self.execute_search_kg_exact(
                self.exact_match_search_tail_entities,
                entity_words_not_visited,
                entity,
                "search_for_tails",
                total_triples
            )

            total_results_dict["matched_entities"] = (
                head_match_triples["matched_entities"]
                | tail_match_triples["matched_entities"]
            )

            # print("len of total triples after exact matching = ", len(total_triples_found))
            return total_triples, total_results_dict

        else:
            return total_triples, total_results_dict
    
    def execute_search_kg_partial(
        self,
        cypher_query: str,
        entity_id: List[str],
        entity: str,
        flag: str,
        total_triples: Set[Tuple[str, str, str]]
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        result_dict = {"matched_entities": {}}
        regex_pattern = r"(?i).*(" + "|".join([re.escape(word.strip()) for word in entity_id]) + r").*"

        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            if results:
                for record in results:
                    head_ent = record["head_entity"].strip().replace("_", " ").lower()
                    relation = record["relation"].strip().replace("_", " ").lower()
                    tail_ent = record["tail_entity"].strip().replace("_", " ").lower()
                    triple = (head_ent, relation, tail_ent)

                    if flag == "search_for_heads" and any(word in tail_ent for word in entity_id):
                        result_dict, total_triples = self._add_kg_results(result_dict, tail_ent, triple, entity, entity_id, total_triples)
                    elif flag == "search_for_tails" and any(word in head_ent for word in entity_id):
                        result_dict, total_triples = self._add_kg_results(result_dict, head_ent, triple, entity, entity_id, total_triples)
                    elif flag == "search_subj_obj" and any(word in relation for word in entity_id):
                        result_dict, total_triples = self._add_kg_results(result_dict, relation, triple, entity, entity_id, total_triples)
                    else:
                        continue
            return result_dict, total_triples
        except Exception as e:
            # print(f"[ERROR - Partial Search] Entity: {entity_id} | Flag: {flag} | Error: {e}")
            return {"matched_entities": {}}, total_triples
        
    def execute_search_kg_exact(
        self,
        cypher_query: str,
        entity_id: List[str],
        entity: str,
        flag: str,
        total_triples: Set[Tuple[str, str, str]]
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        result_dict = {"matched_entities": {}}
        regex_pattern = r"(?i)\\b(" + "|".join([re.escape(word.strip()) for word in entity_id]) + r")\\b"

        try:
            results = self.graph_store.query(cypher_query, {"re": regex_pattern})
            if results:
                for record in results:
                    head_ent = record["head_entity"].strip().replace("_", " ").lower()
                    relation = record["relation"].strip().replace("_", " ").lower()
                    tail_ent = record["tail_entity"].strip().replace("_", " ").lower()
                    triple = (head_ent, relation, tail_ent)

                    if flag == "search_for_heads" and any(word in tail_ent for word in entity_id):
                        result_dict, total_triples = self._add_kg_results(result_dict, tail_ent, triple, entity, entity_id, total_triples)
                    elif flag == "search_for_tails" and any(word in head_ent for word in entity_id):
                        result_dict, total_triples = self._add_kg_results(result_dict, head_ent, triple, entity, entity_id, total_triples)
                    else:
                        continue
            return result_dict, total_triples
        except Exception as e:
            # print(f"[ERROR - Exact Search] Entity: {entity_id} | Flag: {flag} | Error: {e}")
            return {"matched_entities": {}}, total_triples
        
    def _add_kg_results(
        self,
        results: Dict[str, Any],
        matched_entity: str,
        triple: Tuple[str, str, str],
        original_entity: str,
        entity_id: List[str],
        total_triples: Set[Tuple[str, str, str]]
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        """
        Add the matched triple to result dict and categorize for traversal.
        """
        if matched_entity not in results["matched_entities"]:
            results["matched_entities"][matched_entity] = {
                "matched_triples": set(),
                "source": set(),
                "candidate_from_previous": set(),
                "next_candidate": set()
            }

        if triple not in total_triples:
            total_triples.add(triple)

            subj, rel, obj = triple

            results["matched_entities"][matched_entity]["matched_triples"].add(triple)

            for word in entity_id:
                word = word.lower()

                if word in matched_entity:
                    results["matched_entities"][matched_entity]["source"].add(word)
                    results["matched_entities"][matched_entity]["candidate_from_previous"].add(original_entity)

                if word in subj:
                    results["matched_entities"][matched_entity]["next_candidate"].add(obj)
                elif word in obj:
                    results["matched_entities"][matched_entity]["next_candidate"].add(subj)
                elif word in rel:
                    results["matched_entities"][matched_entity]["next_candidate"].add(subj)
                    results["matched_entities"][matched_entity]["next_candidate"].add(obj)

        return results, total_triples
    
    def _remove_stopwords(self, word: str) -> str:
        """
        Removes common stopwords and punctuation from a word.
        Returns the cleaned word or empty string if it's a stopword/punctuation.
        """
        stop_words = set(stopwords.words("english"))
        word = word.strip().lower()
        word = word.translate(str.maketrans("", "", string.punctuation))

        if word in stop_words or not word:
            return ""
        return word


    # --- Entity and Triple Management ---
    def _clean_triple_prune_output(self, output: str) -> Tuple[bool, List[Tuple[str, str, str]]]:
        """
        Extracts high-confidence triples from LLM output based on score ≥ 8.
        Returns a flag and list of triples.
        """
        pattern = r"\*\*\((.*?)\)\*\* \(Score: (\d+)\)|\((.*?)\) \(Score: (\d+)\)"
        matches = re.findall(pattern, output)

        high_score_triples = []
        for triple, score, triple_alt, score_alt in matches:
            raw_triple_str = triple or triple_alt
            raw_score = int(score or score_alt)

            if raw_score >= 8:
                try:
                    # Clean and convert "(a, b, c)" into actual tuple
                    triple_tuple = tuple([x.strip().strip("'\"") for x in raw_triple_str.split(",")])
                    if len(triple_tuple) == 3:
                        high_score_triples.append(triple_tuple)
                except Exception as e:
                    print(f"[clean_triple_prune_output] Failed to parse triple: {raw_triple_str} | Error: {e}")
                    continue

        if not high_score_triples:
            print("no triples found")
            return False, []

        return True, high_score_triples


    def _prune_triples(
        self,
        question: str,
        triples: Set[Tuple[str, str, str]]
    ) -> Set[Tuple[str, str, str]]:
        """
        Prunes KG triples using LLM reasoning based on the input question.
        Returns a list of high-scoring triples.
        """
        print("\n-- prune triples called --")
        print("len of triples received = ", len(triples))

        if len(triples) == 0:
            return []

        triples_list = list(triples)
        total_triples = "\n".join([str(triple) for triple in triples_list[:500]])  # max 500 triples

        # Construct pruning prompt
        prompt = self.prune_triple_prompt + question + "\nKnowledge Triplets:\n" + total_triples + "\nAnswer: "

        print("--Pruning KG relations--")

        # Run with appropriate LLM
        result = self.llm.complete(prompt).text
        
        flag, retrieve_triples_with_scores = self._clean_triple_prune_output(result)
        print("\noutput of clean triples: ", flag, retrieve_triples_with_scores)

        return retrieve_triples_with_scores if flag else []

    # --- Candidate Entity Extraction / Path Traversal ---
    
    def _clean_and_add_single_quotes(self,triple_tuple):
        cleaned_tuple = []
        for item in triple_tuple:
            # Strip whitespace and remove surrounding single quotes
            new_item = item.strip().strip("'")
            cleaned_tuple.append(new_item.lower())
        return tuple(cleaned_tuple)
    
    def _normalize_text(self, text: str) -> str:
        return text.lower().replace("’", "").replace("'", "").strip()

    def _save_candidates_for_next_round(
        self,
        triples_after_pruning: List[Tuple[str, str, str]],
        all_ent_results_dict: Dict,
        new_path_dict: Dict,
        depth: int,
        memory: str,
        path_traversal_memory: str
    ) -> Tuple[Set[str], List[str], Dict, List[Tuple[str, str, str]]]:

        print("\n=== INPUT TO _save_candidates_for_next_round ===")
        print(f"Depth: {depth}")
        print(f"Triples after pruning ({len(triples_after_pruning)}):")
        for t in triples_after_pruning:
            print("  ", t)
        print(f"Memory: {memory.strip()}")
        print(f"Path Traversal Memory: {path_traversal_memory.strip()}")

        candidate_entities = set()
        path_traversed_list = []
        memory_triple_list = []
        triples_after_pruning_list = []
        path_list = []

        if depth != 1:
            memory_list = memory.split('\n')
            memory_triple_list = [item.split(',') for item in memory_list]
            memory_triple_list = [
                [item.strip() for item in sublist if item.strip()]
                for sublist in memory_triple_list
                if any(item.strip() for item in sublist)
            ]

            path_memory_list = path_traversal_memory.split('\n')
            path_list = [item.split('-->') for item in path_memory_list]
            path_list = [
                [item.strip() for item in sublist if item.strip()]
                for sublist in path_list
                if any(item.strip() for item in sublist)
            ]

            for row in path_list:
                if len(row) >= 3:
                    element = row[1]
                    try:
                        triple = eval(row[-1])
                        if element == triple[0]:
                            row.append(triple[2])
                        elif element == triple[2]:
                            row.append(triple[0])
                    except:
                        continue

        for triple_raw in triples_after_pruning:
            if not isinstance(triple_raw, tuple) or len(triple_raw) != 3:
                continue

            cleaned = tuple(self._clean_triple_item(item) for item in triple_raw)
            subj, rel, obj = cleaned
            path_traversed = "NA"

            for entity, data in all_ent_results_dict["matched_entities"].items():
                if isinstance(data['matched_triples'], str):
                    stored_triples = ast.literal_eval(data['matched_triples'])
                else:
                    stored_triples = data['matched_triples']

                source_ent = data.get("source", "NA")
                candidate_from_previous = data.get("candidate_from_previous", "NA")

                new_stored_triples = [
                    tuple(
                        item.replace('’', "").strip().strip("'").lower()
                        for item in tup
                    )
                    for tup in stored_triples
                ]

                for single_triple in new_stored_triples:
                    match = all(
                        self._remove_stopwords(item1) == self._remove_stopwords(item2)
                        for item1, item2 in zip(cleaned, single_triple)
                    )

                    if match:
                        if entity == subj:
                            candidate_entities.add(obj)
                        elif entity == obj:
                            candidate_entities.add(subj)
                        elif entity == rel:
                            candidate_entities.update([subj, obj])

                        if cleaned not in new_path_dict:
                            new_path_dict[cleaned] = (candidate_from_previous, entity)

                            if depth != 1:
                                for row in path_list:
                                    if len(row) > 3 and entity == row[3]:
                                        path_traversed = f"{row[0]} --> {row[1]} --> {row[2]} --> {row[3]} --> {cleaned}"
                                    else:
                                        path_traversed = f"{candidate_from_previous} --> {source_ent} --> {entity} --> {cleaned}"
                                    if path_traversed not in path_traversed_list:
                                        path_traversed_list.append(path_traversed)
                                    triples_after_pruning_list.append(cleaned)
                            else:
                                path_traversed = f"{candidate_from_previous} --> {entity} --> {cleaned}"
                                if path_traversed not in path_traversed_list:
                                    path_traversed_list.append(path_traversed)
                                triples_after_pruning_list.append(cleaned)
                        else:
                            continue

        # print("\n-- SAVE CANDIDATES OUTPUT --")
        # print(f"Candidate Entities ({len(candidate_entities)}): {candidate_entities}")
        # print(f"Path Traversed ({len(path_traversed_list)}):")
        # for path in path_traversed_list:
        #     print(f"  {path}")
        # print(f"New Path Dict (keys): {list(new_path_dict.keys())}")
        # print(f"Pruned Triple List ({len(triples_after_pruning_list)}): {triples_after_pruning_list}")

        return candidate_entities, path_traversed_list, new_path_dict, triples_after_pruning_list

    
    # --- Reasoning + Reasoning without KG ---
    
    def _extract_answer(self, text: str) -> str:
        """
        Extracts the answer from a string containing {...}.
        Returns the inner content or an empty string if not found.
        """
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1:
            return text[start_index + 1:end_index].strip()
        return ""

    def _if_true(self, prompt: str) -> bool:
        """
        Checks if the given string is equivalent to a 'yes' response.
        Ignores case and whitespace.
        """
        return prompt.lower().strip().replace(" ", "") == "yes"
    
    def _reasoning(
        self,
        question: str,
        cluster_chain_of_entities: List[Tuple[str, str, str]],
        clue: str
    ) -> Tuple[bool, str]:
        """
        Performs LLM-based reasoning over current triples and clues.
        Returns a flag indicating if reasoning should stop, and the raw response.
        """
        print("\n--Running LLM Reasoning--")

        # Construct the prompt
        prompt = self.reasoning_prompt + question
        chain_prompt = "\n".join([str(triple[0]) if isinstance(triple, tuple) else str(triple) for triple in cluster_chain_of_entities])
        prompt += "\nClues: " + clue + "\nKnowledge Triplets:\n" + chain_prompt + '\nAnswer: '

        # Call the appropriate LLM
        response = self.llm.complete(prompt).text
        
        # Analyze response
        result = self._extract_answer(response)
        if self._if_true(result):
            return True, response
        else:
            return False, response
        
        
    def _clean_reasoning_without_kg_output(self, output: str) -> List[str]:
        """
        Extracts all entities wrapped in `{}` from the reasoning output.
        """
        entity_pattern = r"\{(.*?)\}"
        entities = re.findall(entity_pattern, output)
        return [entity.strip() for entity in entities if entity.strip()]
    
    def _reasoning_without_kg(self, question: str) -> str:
        """
        Fallback LLM reasoning when no KG triples are available.
        Returns the raw LLM response.
        """
        print("\n--Running LLM Reasoning WITHOUT KG--")
        prompt = self.reasoning_without_kg_prompt + question + '\nAnswer: '

        response = self.llm.complete(prompt).text

        return response

    # --- Memory Update ---
    
    def _clean_triple_item(self, part: str) -> str:
        """
        Strips whitespace, punctuation and standardizes triple parts.
        """
        return part.strip().strip("'\"").replace("’", "").lower()

    
    def _update_memory(
        self,
        memory: str,
        triples_list: List[Tuple[str, str, str]]
    ) -> str:
        """
        Merges new triples into memory while removing duplicates.
        Each triple is stored as a comma-separated line.
        """
        memory_triples = memory.strip().split("\n") if memory.strip() else []

        seen_triples = set()
        unique_triples = []

        # Clean existing memory triples
        for triple in memory_triples:
            cleaned = tuple(self._clean_triple_item(part) for part in triple.split(","))
            if cleaned not in seen_triples:
                seen_triples.add(cleaned)
                unique_triples.append(cleaned)

        # Clean new triples and add
        for triple in triples_list:
            cleaned = tuple(self._clean_triple_item(part) for part in triple)
            if cleaned not in seen_triples:
                seen_triples.add(cleaned)
                unique_triples.append(cleaned)

        updated_memory = "\n".join(", ".join(triple) for triple in unique_triples)
        return updated_memory

    
    # --- Cleaning Outputs ---

    def _clean_reasoning_output(self, output: str) -> Tuple[str, List[str]]:
        """
        Splits reasoning output into remaining explanation and extracted entities.
        Expects format like: {Yes} Some explanation... {Entity1}, {Entity2}
        """
        split_pattern = r"^{(Yes|No)}\s*(.*)"
        match = re.match(split_pattern, output.strip())

        remaining_text = output.strip()
        if match:
            remaining_text = match.group(2)

        entity_pattern = r"\{(.*?)\}"
        entities = re.findall(entity_pattern, remaining_text)

        return remaining_text.strip(), [e.strip() for e in entities if e.strip()]

    def _clean_reasoning_without_kg_output(self, output: str) -> List[str]:
        """
        Extracts all entities wrapped in `{}` from the fallback reasoning output.
        """
        entity_pattern = r"\{(.*?)\}"
        entities = re.findall(entity_pattern, output)
        return [e.strip() for e in entities if e.strip()]
    
    def _update_visited_entities_and_triples(
        self,
        retrieved_triples: Set[Tuple[str, str, str]],
        entity: str,
        retrieved_ent_results: Dict,
        all_entity_triples: Set[Tuple[str, str, str]],
        visited_entities: Set[str],
        all_ent_results_dict: Dict[str, Any]
    ) -> Tuple[Set[Tuple[str, str, str]], Set[str]]:
        """
        Updates visited_entities and all_entity_triples using newly retrieved triples and matched entity data.
        """
        # Add new triples to global triple set
        all_entity_triples.update(retrieved_triples)

        # Mark the current entity's words as visited
        visited_entities.update(entity.lower().split())

        # Merge retrieved entity results into the global dict
        for key, value in retrieved_ent_results.get("matched_entities", {}).items():
            if key in all_ent_results_dict["matched_entities"]:
                all_ent_results_dict["matched_entities"][key]["matched_triples"].update(value.get("matched_triples", set()))
                all_ent_results_dict["matched_entities"][key]["source"].update(value.get("source", set()))
            else:
                all_ent_results_dict["matched_entities"][key] = value

        # Mark all entity terms in matched results as visited (after stopword removal)
        for entity_term in all_ent_results_dict["matched_entities"].keys():
            if (new_word := self._remove_stopwords(entity_term)):
                visited_entities.update(new_word.lower().split())

        return all_entity_triples, visited_entities