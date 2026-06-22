from __future__ import annotations

import ast
import re
import string
from typing import Any, Dict, List, Literal, Set, Tuple

from kgcare.graph_store import Neo4jTripleStore
from kgcare.llm import OpenAIChatClient
from kgcare.prompts import KEYWORD_EXTRACT_PROMPT, PRUNE_TRIPLE_PROMPT, REASONING_PROMPT, REASONING_WITHOUT_KG_PROMPT
from kgcare.schemas import (
    ClueDecision,
    PruneDecision,
    ReasoningDecision,
    RetrievalResult,
    TraversalOutputMode,
    TraversalStepTrace,
    TraversalTrace,
    TripleCandidate,
)
from kgcare.vector_store import FaissChunkStore

DEFAULT_NODE_SCORE = 1000.0

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def _load_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return FALLBACK_STOPWORDS


STOPWORDS = _load_stopwords()


def extract_keywords_given_response(response: str) -> list[str]:
    if "KEYWORDS:" in response:
        response = response.split("KEYWORDS:", 1)[1]
    response = response.strip().strip("'").strip('"')
    return [item.strip().strip("'\"") for item in response.split(",") if item.strip()]


class KGCaReTraversal:
    # Partial match queries. Relation search reads the extracted relation property
    # rather than Neo4j's generic RELATES_TO type, preserving effective triples.
    partial_match_search_head_entities = """
    MATCH (headEntity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
      -[relation:RELATES_TO {dataset: $dataset, index_name: $index_name}]->
      (entity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
    WHERE toLower(entity.id) =~ $re
    RETURN headEntity.id AS head_entity, relation.relation AS relation, entity.id AS tail_entity
    """

    partial_match_search_tail_entities = """
    MATCH (entity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
      -[relation:RELATES_TO {dataset: $dataset, index_name: $index_name}]->
      (tailEntity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
    WHERE toLower(entity.id) =~ $re
    RETURN entity.id AS head_entity, relation.relation AS relation, tailEntity.id AS tail_entity
    """

    partial_match_search_rel_entities = """
    MATCH (headEntity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
      -[relation:RELATES_TO {dataset: $dataset, index_name: $index_name}]->
      (tailEntity:KGCaReEntity {dataset: $dataset, index_name: $index_name})
    WHERE toLower(relation.relation) =~ $re
    RETURN headEntity.id AS head_entity, relation.relation AS relation, tailEntity.id AS tail_entity
    """

    exact_match_search_head_entities = partial_match_search_head_entities
    exact_match_search_tail_entities = partial_match_search_tail_entities

    def __init__(
        self,
        graph_store: Neo4jTripleStore,
        llm: OpenAIChatClient,
        max_depth: int = 3,
        max_entities: int = 20,
        max_keywords_per_query: int = 20,
        traversal_output_mode: TraversalOutputMode = "structured",
        include_trace: bool = True,
        structured_fallback: bool = True,
    ) -> None:
        self.graph_store = graph_store
        self.llm = llm
        self.max_depth = max_depth
        self.max_entities = max_entities
        self.max_keywords_per_query = max_keywords_per_query
        self.traversal_output_mode = traversal_output_mode
        self.include_trace = include_trace
        self.structured_fallback = structured_fallback

    def retrieve(self, question_id: str, query_str: str, topic_entities: list[str] | None = None) -> RetrievalResult:
        kg_context, triples, answer_hint, trace = self._retrieve_kg_context(query_str, topic_entities or [])
        return RetrievalResult(
            question_id=question_id,
            question=query_str,
            mode="kg",
            kg_context=kg_context,
            triples=[list(triple) for triple in triples],
            answer_hint=answer_hint,
            trace=trace if self.include_trace else None,
        )

    def _retrieve_kg_context(
        self,
        query_str: str,
        topic_entities: list[str],
    ) -> tuple[str, list[tuple[str, str, str]], str, TraversalTrace | None]:
        if not topic_entities:
            topic_entities = self._get_keywords(query_str)
        if not topic_entities:
            raise ValueError("No topic entities found in query.")
        trace = TraversalTrace(output_mode=self.traversal_output_mode, initial_entities=list(topic_entities))

        visited_entities: Set[str] = set()
        clue_memory = ""
        path_traversal_memory = ""
        memory = ""
        clue_entities: list[str] = []
        new_path_dict: dict[tuple[str, str, str], Any] = {}
        depth = 1
        answer_hint = ""

        while depth <= self.max_depth and (topic_entities or clue_entities):
            all_entity_triples: set[tuple[str, str, str]] = set()
            all_ent_results_dict: dict[str, Any] = {"matched_entities": {}}
            searched_topic_entities: list[str] = []
            searched_clue_entities: list[str] = []
            partial_searches = 0
            exact_searches = 0
            clue_searches = 0

            for entity in topic_entities:
                entity = entity.replace("-", " ")
                searched_topic_entities.append(entity)
                if depth == 1:
                    partial_searches += 1
                    retrieved_triples, retrieved_ent_results = self._search_kg_partial(
                        entity, visited_entities, all_entity_triples
                    )
                else:
                    exact_searches += 1
                    retrieved_triples, retrieved_ent_results = self._search_kg_exact(
                        entity, visited_entities, all_entity_triples
                    )
                all_entity_triples, visited_entities = self._update_visited_entities_and_triples(
                    retrieved_triples,
                    entity,
                    retrieved_ent_results,
                    all_entity_triples,
                    visited_entities,
                    all_ent_results_dict,
                )

            for entity in clue_entities:
                entity = entity.replace("-", " ")
                searched_clue_entities.append(entity)
                clue_searches += 1
                retrieved_triples, retrieved_ent_results = self._search_kg_partial(
                    entity, visited_entities, all_entity_triples
                )
                all_entity_triples, visited_entities = self._update_visited_entities_and_triples(
                    retrieved_triples,
                    entity,
                    retrieved_ent_results,
                    all_entity_triples,
                    visited_entities,
                    all_ent_results_dict,
                )

            pruned_triples, selected_candidates = self._prune_triples(query_str, all_entity_triples)
            candidate_entities, path_traversed, new_path_dict, pruned_triple_list = self._save_candidates_for_next_round(
                pruned_triples,
                all_ent_results_dict,
                new_path_dict,
                depth,
                memory,
                path_traversal_memory,
            )

            topic_entities = list(set(candidate_entities - visited_entities))[:50]
            memory = self._update_memory(memory, pruned_triple_list)
            path_traversal_memory += "\n" + "\n".join(path_traversed)
            step_trace = TraversalStepTrace(
                depth=depth,
                topic_entities=list(searched_topic_entities),
                clue_entities=list(searched_clue_entities),
                searched_topic_entities=searched_topic_entities,
                searched_clue_entities=searched_clue_entities,
                partial_searches=partial_searches,
                exact_searches=exact_searches,
                clue_searches=clue_searches,
                candidate_triples_found=len(all_entity_triples),
                selected_triples=selected_candidates,
                next_topic_entities=list(topic_entities),
            )

            if pruned_triples:
                reasoning = self._reasoning(query_str, pruned_triple_list, clue_memory)
                step_trace.reasoning_sufficient = reasoning.sufficient
                step_trace.reasoning_answer = reasoning.answer
                step_trace.reasoning_rationale = reasoning.rationale
                if reasoning.sufficient:
                    answer_hint = reasoning.answer
                    step_trace.stop_reason = "sufficient"
                    trace.steps.append(step_trace)
                    break
                clue_memory += "\n" + reasoning.rationale
                clue_entities = list(set(reasoning.clue_entities) - visited_entities)
                step_trace.next_clue_entities = list(clue_entities)
            else:
                clue_decision = self._reasoning_without_kg(query_str)
                clue_memory += "\n" + clue_decision.rationale
                clue_entities = list(set(clue_decision.clue_entities) - visited_entities)
                step_trace.reasoning_sufficient = False
                step_trace.reasoning_rationale = clue_decision.rationale
                step_trace.next_clue_entities = list(clue_entities)
                if not clue_entities and not topic_entities:
                    step_trace.stop_reason = "no_candidates"
                    trace.steps.append(step_trace)
                    break

            trace.steps.append(step_trace)
            visited_entities.update(topic_entities)
            visited_entities.update(clue_entities)
            depth += 1

        triples = []
        for line in memory.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
        final_text = f"Answer: {answer_hint}\n\nKG context:\n{memory}\n" if answer_hint else f"KG context:\n{memory}\n"
        trace.final_answer_hint = answer_hint
        return final_text, triples, answer_hint, trace

    def _get_keywords(self, query_str: str) -> List[str]:
        prompt = KEYWORD_EXTRACT_PROMPT.format(max_keywords=self.max_keywords_per_query, question=query_str)
        response = self.llm.chat_text(prompt)
        return extract_keywords_given_response(response)

    def _search_kg_partial(
        self,
        entity: str,
        visited_entities: Set[str],
        total_triples: Set[Tuple[str, str, str]],
    ) -> Tuple[Set[Tuple[str, str, str]], Dict[str, Any]]:
        total_results_dict: dict[str, Any] = {"matched_entities": {}}
        entity_words = entity.split()
        new_entity_words = [
            new_word.replace(".", "").lower()
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
                total_triples,
            )
            tail_match_triples, total_triples_found = self.execute_search_kg_partial(
                self.partial_match_search_tail_entities,
                entity_words_not_visited,
                entity,
                "search_for_tails",
                total_triples,
            )
            rel_match_triples, total_triples_found = self.execute_search_kg_partial(
                self.partial_match_search_rel_entities,
                entity_words_not_visited,
                entity,
                "search_subj_obj",
                total_triples,
            )
            total_results_dict["matched_entities"] = (
                head_match_triples["matched_entities"]
                | tail_match_triples["matched_entities"]
                | rel_match_triples["matched_entities"]
            )
            return total_triples, total_results_dict
        return total_triples, total_results_dict

    def _search_kg_exact(
        self,
        entity: str,
        visited_entities: Set[str],
        total_triples: Set[Tuple[str, str, str]],
    ) -> Tuple[Set[Tuple[str, str, str]], Dict[str, Any]]:
        total_results_dict: dict[str, Any] = {"matched_entities": {}}
        entity_words = entity.split()
        new_entity_words = [
            new_word.replace(".", "").lower()
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
                total_triples,
            )
            tail_match_triples, total_triples_found = self.execute_search_kg_exact(
                self.exact_match_search_tail_entities,
                entity_words_not_visited,
                entity,
                "search_for_tails",
                total_triples,
            )
            total_results_dict["matched_entities"] = (
                head_match_triples["matched_entities"] | tail_match_triples["matched_entities"]
            )
            return total_triples, total_results_dict
        return total_triples, total_results_dict

    def execute_search_kg_partial(
        self,
        cypher_query: str,
        entity_id: List[str],
        entity: str,
        flag: str,
        total_triples: Set[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        result_dict: dict[str, Any] = {"matched_entities": {}}
        regex_pattern = r"(?i).*(" + "|".join([re.escape(word.strip()) for word in entity_id]) + r").*"
        try:
            results = self.graph_store.query(
                cypher_query,
                {
                    "re": regex_pattern,
                    "dataset": self.graph_store.dataset,
                    "index_name": self.graph_store.index_name,
                },
            )
            for record in results:
                head_ent = record["head_entity"].strip().replace("_", " ").lower()
                relation = record["relation"].strip().replace("_", " ").lower()
                tail_ent = record["tail_entity"].strip().replace("_", " ").lower()
                triple = (head_ent, relation, tail_ent)
                if flag == "search_for_heads" and any(word in tail_ent for word in entity_id):
                    result_dict, total_triples = self._add_kg_results(
                        result_dict, tail_ent, triple, entity, entity_id, total_triples
                    )
                elif flag == "search_for_tails" and any(word in head_ent for word in entity_id):
                    result_dict, total_triples = self._add_kg_results(
                        result_dict, head_ent, triple, entity, entity_id, total_triples
                    )
                elif flag == "search_subj_obj" and any(word in relation for word in entity_id):
                    result_dict, total_triples = self._add_kg_results(
                        result_dict, relation, triple, entity, entity_id, total_triples
                    )
            return result_dict, total_triples
        except Exception:
            return {"matched_entities": {}}, total_triples

    def execute_search_kg_exact(
        self,
        cypher_query: str,
        entity_id: List[str],
        entity: str,
        flag: str,
        total_triples: Set[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        result_dict: dict[str, Any] = {"matched_entities": {}}
        regex_pattern = r"(?i)\\b(" + "|".join([re.escape(word.strip()) for word in entity_id]) + r")\\b"
        try:
            results = self.graph_store.query(
                cypher_query,
                {
                    "re": regex_pattern,
                    "dataset": self.graph_store.dataset,
                    "index_name": self.graph_store.index_name,
                },
            )
            for record in results:
                head_ent = record["head_entity"].strip().replace("_", " ").lower()
                relation = record["relation"].strip().replace("_", " ").lower()
                tail_ent = record["tail_entity"].strip().replace("_", " ").lower()
                triple = (head_ent, relation, tail_ent)
                if flag == "search_for_heads" and any(word in tail_ent for word in entity_id):
                    result_dict, total_triples = self._add_kg_results(
                        result_dict, tail_ent, triple, entity, entity_id, total_triples
                    )
                elif flag == "search_for_tails" and any(word in head_ent for word in entity_id):
                    result_dict, total_triples = self._add_kg_results(
                        result_dict, head_ent, triple, entity, entity_id, total_triples
                    )
            return result_dict, total_triples
        except Exception:
            return {"matched_entities": {}}, total_triples

    def _add_kg_results(
        self,
        results: Dict[str, Any],
        matched_entity: str,
        triple: Tuple[str, str, str],
        original_entity: str,
        entity_id: List[str],
        total_triples: Set[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, Any], Set[Tuple[str, str, str]]]:
        if matched_entity not in results["matched_entities"]:
            results["matched_entities"][matched_entity] = {
                "matched_triples": set(),
                "source": set(),
                "candidate_from_previous": set(),
                "next_candidate": set(),
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
        word = word.strip().lower()
        word = word.translate(str.maketrans("", "", string.punctuation))
        if word in STOPWORDS or not word:
            return ""
        return word

    def _clean_triple_prune_output(self, output: str) -> Tuple[bool, List[Tuple[str, str, str]]]:
        parsed = self._clean_triple_prune_output_with_scores(output)
        high_score_triples = [triple for triple, score, _ in parsed if score >= 8]
        if not high_score_triples:
            return False, []
        return True, high_score_triples

    def _clean_triple_prune_output_with_scores(self, output: str) -> List[Tuple[Tuple[str, str, str], float, str]]:
        pattern = r"\*\*\((.*?)\)\*\* \(Score: (\d+)\)|\((.*?)\) \(Score: (\d+)\)"
        matches = re.findall(pattern, output)
        parsed: list[tuple[tuple[str, str, str], float, str]] = []
        for triple, score, triple_alt, score_alt in matches:
            raw_triple_str = triple or triple_alt
            raw_score = float(score or score_alt)
            triple_tuple = tuple([x.strip().strip("'\"") for x in raw_triple_str.split(",")])
            if len(triple_tuple) == 3:
                parsed.append((triple_tuple, raw_score, ""))  # type: ignore[arg-type]
        return parsed

    def _candidate_from_triple(
        self,
        index: int,
        triple: Tuple[str, str, str],
        score: float | None = None,
        rationale: str = "",
    ) -> TripleCandidate:
        return TripleCandidate(index=index, triple=[triple[0], triple[1], triple[2]], score=score, rationale=rationale)

    def _indexed_triples_text(self, triples: List[Tuple[str, str, str]]) -> str:
        return "\n".join(f"{index}. ({head}, {relation}, {tail})" for index, (head, relation, tail) in enumerate(triples))

    def _prune_triples(
        self,
        question: str,
        triples: Set[Tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str, str]], list[TripleCandidate]]:
        if len(triples) == 0:
            return [], []
        triples_list = sorted(triples)[:500]
        if self.traversal_output_mode == "structured":
            try:
                return self._prune_triples_structured(question, triples_list)
            except Exception:
                if not self.structured_fallback:
                    raise
        return self._prune_triples_text(question, triples_list)

    def _prune_triples_text(
        self,
        question: str,
        triples_list: List[Tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str, str]], list[TripleCandidate]]:
        total_triples = "\n".join([str(triple) for triple in triples_list])
        prompt = PRUNE_TRIPLE_PROMPT + question + "\nKnowledge Triplets:\n" + total_triples + "\nAnswer: "
        result = self.llm.chat_text(prompt)
        parsed = self._clean_triple_prune_output_with_scores(result)
        selected_triples: list[tuple[str, str, str]] = []
        selected_candidates: list[TripleCandidate] = []
        for triple, score, rationale in parsed:
            if score < 8:
                continue
            selected_triples.append(triple)
            index = triples_list.index(triple) if triple in triples_list else -1
            selected_candidates.append(self._candidate_from_triple(index, triple, score=score, rationale=rationale))
        return selected_triples, selected_candidates

    def _prune_triples_structured(
        self,
        question: str,
        triples_list: List[Tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str, str]], list[TripleCandidate]]:
        prompt = (
            "Select the knowledge triples that are most useful for answering the question.\n"
            "Return only indexes from the provided list. Score each selected triple from 0 to 10.\n"
            "Only select a triple when its score is at least 8.\n\n"
            f"Question:\n{question}\n\n"
            "Indexed Knowledge Triples:\n"
            f"{self._indexed_triples_text(triples_list)}"
        )
        decision, _ = self.llm.structured(
            PruneDecision,
            [
                {
                    "role": "system",
                    "content": "Return structured pruning decisions for KGCaRe graph traversal.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        selected_triples: list[tuple[str, str, str]] = []
        selected_candidates: list[TripleCandidate] = []
        seen_indexes: set[int] = set()
        for selection in decision.selections:
            if selection.score < 8 or selection.index in seen_indexes:
                continue
            if selection.index < 0 or selection.index >= len(triples_list):
                continue
            seen_indexes.add(selection.index)
            triple = triples_list[selection.index]
            selected_triples.append(triple)
            selected_candidates.append(
                self._candidate_from_triple(
                    selection.index,
                    triple,
                    score=selection.score,
                    rationale=selection.rationale or decision.rationale,
                )
            )
        return selected_triples, selected_candidates

    def _save_candidates_for_next_round(
        self,
        triples_after_pruning: List[Tuple[str, str, str]],
        all_ent_results_dict: Dict,
        new_path_dict: Dict,
        depth: int,
        memory: str,
        path_traversal_memory: str,
    ) -> Tuple[Set[str], List[str], Dict, List[Tuple[str, str, str]]]:
        candidate_entities: set[str] = set()
        path_traversed_list: list[str] = []
        triples_after_pruning_list: list[tuple[str, str, str]] = []
        path_list: list[list[str]] = []

        if depth != 1:
            path_memory_list = path_traversal_memory.split("\n")
            path_list = [item.split("-->") for item in path_memory_list]
            path_list = [
                [item.strip() for item in sublist if item.strip()]
                for sublist in path_list
                if any(item.strip() for item in sublist)
            ]
            for row in path_list:
                if len(row) >= 3:
                    element = row[1]
                    try:
                        triple = ast.literal_eval(row[-1])
                        if element == triple[0]:
                            row.append(triple[2])
                        elif element == triple[2]:
                            row.append(triple[0])
                    except Exception:
                        continue

        for triple_raw in triples_after_pruning:
            if not isinstance(triple_raw, tuple) or len(triple_raw) != 3:
                continue
            cleaned = tuple(self._clean_triple_item(item) for item in triple_raw)
            subj, rel, obj = cleaned
            for entity, data in all_ent_results_dict["matched_entities"].items():
                if isinstance(data["matched_triples"], str):
                    stored_triples = ast.literal_eval(data["matched_triples"])
                else:
                    stored_triples = data["matched_triples"]
                source_ent = data.get("source", "NA")
                candidate_from_previous = data.get("candidate_from_previous", "NA")
                new_stored_triples = [
                    tuple(item.replace("’", "").strip().strip("'").lower() for item in tup)
                    for tup in stored_triples
                ]
                for single_triple in new_stored_triples:
                    match = all(
                        self._remove_stopwords(item1) == self._remove_stopwords(item2)
                        for item1, item2 in zip(cleaned, single_triple)
                    )
                    if not match:
                        continue
                    if entity == subj:
                        candidate_entities.add(obj)
                    elif entity == obj:
                        candidate_entities.add(subj)
                    elif entity == rel:
                        candidate_entities.update([subj, obj])
                    if cleaned in new_path_dict:
                        continue
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
        return candidate_entities, path_traversed_list, new_path_dict, triples_after_pruning_list

    def _extract_answer(self, text: str) -> str:
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1:
            return text[start_index + 1 : end_index].strip()
        return ""

    def _if_true(self, prompt: str) -> bool:
        return prompt.lower().strip().replace(" ", "") == "yes"

    def _reasoning(
        self,
        question: str,
        cluster_chain_of_entities: List[Tuple[str, str, str]],
        clue: str,
    ) -> ReasoningDecision:
        if self.traversal_output_mode == "structured":
            try:
                return self._reasoning_structured(question, cluster_chain_of_entities, clue)
            except Exception:
                if not self.structured_fallback:
                    raise
        return self._reasoning_text(question, cluster_chain_of_entities, clue)

    def _reasoning_text(
        self,
        question: str,
        cluster_chain_of_entities: List[Tuple[str, str, str]],
        clue: str,
    ) -> ReasoningDecision:
        prompt = REASONING_PROMPT + question
        chain_prompt = "\n".join([str(triple) for triple in cluster_chain_of_entities])
        prompt += "\nClues: " + clue + "\nKnowledge Triplets:\n" + chain_prompt + "\nAnswer: "
        response = self.llm.chat_text(prompt)
        result = self._extract_answer(response)
        if self._if_true(result):
            return ReasoningDecision(sufficient=True, answer=result, clue_entities=[], rationale=response)
        clues, possible_entities = self._clean_reasoning_output(response)
        return ReasoningDecision(
            sufficient=False,
            answer="",
            clue_entities=possible_entities,
            rationale=clues,
        )

    def _reasoning_structured(
        self,
        question: str,
        cluster_chain_of_entities: List[Tuple[str, str, str]],
        clue: str,
    ) -> ReasoningDecision:
        chain_prompt = "\n".join(
            f"({head}, {relation}, {tail})" for head, relation, tail in cluster_chain_of_entities
        )
        prompt = (
            REASONING_PROMPT
            + question
            + "\nClues: "
            + clue
            + "\nKnowledge Triplets:\n"
            + chain_prompt
            + "\nReturn a structured decision. If sufficient is true, put the answer in answer. "
            "If sufficient is false, put additional search entities in clue_entities."
        )
        decision, _ = self.llm.structured(
            ReasoningDecision,
            [
                {
                    "role": "system",
                    "content": "Return structured KGCaRe traversal sufficiency decisions using only the provided clues and triples.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return decision

    def _reasoning_without_kg(self, question: str) -> ClueDecision:
        if self.traversal_output_mode == "structured":
            try:
                return self._reasoning_without_kg_structured(question)
            except Exception:
                if not self.structured_fallback:
                    raise
        return self._reasoning_without_kg_text(question)

    def _reasoning_without_kg_text(self, question: str) -> ClueDecision:
        prompt = REASONING_WITHOUT_KG_PROMPT + question + "\nAnswer: "
        response = self.llm.chat_text(prompt)
        return ClueDecision(clue_entities=self._clean_reasoning_without_kg_output(response), rationale=response)

    def _reasoning_without_kg_structured(self, question: str) -> ClueDecision:
        prompt = (
            REASONING_WITHOUT_KG_PROMPT
            + question
            + "\nReturn structured clue entities needed for the next KG search."
        )
        decision, _ = self.llm.structured(
            ClueDecision,
            [
                {
                    "role": "system",
                    "content": "Return structured KGCaRe clue entities for graph traversal.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return decision

    def _clean_triple_item(self, part: str) -> str:
        return part.strip().strip("'\"").replace("’", "").lower()

    def _update_memory(self, memory: str, triples_list: List[Tuple[str, str, str]]) -> str:
        memory_triples = memory.strip().split("\n") if memory.strip() else []
        seen_triples: set[tuple[str, ...]] = set()
        unique_triples: list[tuple[str, ...]] = []
        for triple in memory_triples:
            cleaned = tuple(self._clean_triple_item(part) for part in triple.split(","))
            if cleaned not in seen_triples:
                seen_triples.add(cleaned)
                unique_triples.append(cleaned)
        for triple in triples_list:
            cleaned = tuple(self._clean_triple_item(part) for part in triple)
            if cleaned not in seen_triples:
                seen_triples.add(cleaned)
                unique_triples.append(cleaned)
        return "\n".join(", ".join(triple) for triple in unique_triples)

    def _clean_reasoning_output(self, output: str) -> Tuple[str, List[str]]:
        split_pattern = r"^{(Yes|No)}\s*(.*)"
        match = re.match(split_pattern, output.strip())
        remaining_text = output.strip()
        if match:
            remaining_text = match.group(2)
        entity_pattern = r"\{(.*?)\}"
        entities = re.findall(entity_pattern, remaining_text)
        return remaining_text.strip(), [entity.strip() for entity in entities if entity.strip()]

    def _clean_reasoning_without_kg_output(self, output: str) -> List[str]:
        entity_pattern = r"\{(.*?)\}"
        entities = re.findall(entity_pattern, output)
        return [entity.strip() for entity in entities if entity.strip()]

    def _update_visited_entities_and_triples(
        self,
        retrieved_triples: Set[Tuple[str, str, str]],
        entity: str,
        retrieved_ent_results: Dict,
        all_entity_triples: Set[Tuple[str, str, str]],
        visited_entities: Set[str],
        all_ent_results_dict: Dict[str, Any],
    ) -> Tuple[Set[Tuple[str, str, str]], Set[str]]:
        all_entity_triples.update(retrieved_triples)
        visited_entities.update(entity.lower().split())
        for key, value in retrieved_ent_results.get("matched_entities", {}).items():
            if key in all_ent_results_dict["matched_entities"]:
                all_ent_results_dict["matched_entities"][key]["matched_triples"].update(
                    value.get("matched_triples", set())
                )
                all_ent_results_dict["matched_entities"][key]["source"].update(value.get("source", set()))
            else:
                all_ent_results_dict["matched_entities"][key] = value
        for entity_term in all_ent_results_dict["matched_entities"].keys():
            if new_word := self._remove_stopwords(entity_term):
                visited_entities.update(new_word.lower().split())
        return all_entity_triples, visited_entities


class HybridKGCaReRetriever:
    def __init__(
        self,
        vector_store: FaissChunkStore | None,
        kg_retriever: KGCaReTraversal | None,
        vector_top_k: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.kg_retriever = kg_retriever
        self.vector_top_k = vector_top_k

    def retrieve(
        self,
        question_id: str,
        question: str,
        mode: str,
        topic_entities: list[str] | None = None,
    ) -> RetrievalResult:
        vector_context = ""
        vector_chunk_ids: list[str] = []
        kg_context = ""
        triples: list[list[str]] = []
        answer_hint = ""
        trace = None

        if mode in {"hybrid", "vector"} and self.vector_store is not None and self.vector_top_k > 0:
            vector_hits = self.vector_store.search(question, top_k=self.vector_top_k)
            vector_context = "\n\n".join(hit.text for hit, _ in vector_hits)
            vector_chunk_ids = [hit.chunk_id for hit, _ in vector_hits]

        if mode in {"hybrid", "kg"} and self.kg_retriever is not None:
            kg_result = self.kg_retriever.retrieve(question_id, question, topic_entities=topic_entities)
            kg_context = kg_result.kg_context
            triples = kg_result.triples
            answer_hint = kg_result.answer_hint
            trace = kg_result.trace

        return RetrievalResult(
            question_id=question_id,
            question=question,
            mode=mode,  # type: ignore[arg-type]
            vector_context=vector_context,
            kg_context=kg_context,
            vector_chunk_ids=vector_chunk_ids,
            triples=triples,
            answer_hint=answer_hint,
            trace=trace,
        )
