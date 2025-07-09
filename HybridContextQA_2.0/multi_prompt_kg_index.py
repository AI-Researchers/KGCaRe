from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.prompts import BasePromptTemplate
from typing import List, Tuple, Optional

class MultiPromptKnowledgeGraphIndex(KnowledgeGraphIndex):
    """Custom Knowledge Graph Index with extended functionalities."""

    def __init__(
        self,
        *args,
        kg_triplet_extract_templates: Optional[List[BasePromptTemplate]] = None,
        **kwargs
    ):
        # Accept a list of templates and pass the rest to the base class
        # Ensure the custom attribute is initialized
        self.kg_triplet_extract_templates = (
            kg_triplet_extract_templates or [self.kg_triplet_extract_templates]
        )
        
        super().__init__(*args, **kwargs)

    def _llm_extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract triplets using multiple prompts sequentially."""
        
        for index , template in enumerate(self.kg_triplet_extract_templates):
            if index == 0:
                response = self._llm.predict(
                    template,
                    text=text
                )
            else:
                response = self._llm.predict(
                    template,
                    text=text,
                    intermediate_extracted=intermediate_extracted
                )
            intermediate_extracted = response

        return self._parse_triplet_response(
            intermediate_extracted, max_length=self._max_object_length
        )
        
    @staticmethod
    def _parse_triplet_response(
            response: str, max_length: int = 512
        ) -> List[Tuple[str, str, str]]:
            knowledge_strs = response.strip().split("\n")
            results = []
            for text in knowledge_strs:
                if "(" not in text or ")" not in text or text.index(")") < text.index("("):
                    # skip empty lines and non-triplets
                    continue
                triplet_part = text[text.index("(") + 1 : text.index(")")]
                tokens = triplet_part.split(",")
                if len(tokens) != 3:
                    continue

                if any(len(s.encode("utf-8")) > max_length for s in tokens):
                    # We count byte-length instead of len() for UTF-8 chars,
                    # will skip if any of the tokens are too long.
                    # This is normally due to a poorly formatted triplet
                    # extraction, in more serious KG building cases
                    # we'll need NLP models to better extract triplets.
                    continue

                subj, pred, obj = map(str.strip, tokens)
                if not subj or not pred or not obj:
                    # skip partial triplets
                    continue

                # Strip double quotes and Capitalize triplets for disambiguation
                subj, pred, obj = (
                    entity.strip('"').capitalize() for entity in [subj, pred, obj]
                )
                
                # Remove leading and trailing quotes and whitespace
                pred = pred.strip("'").strip()
                
                # Replace invalid characters with spaces
                invalid_chars = ['&', '*', ':', 'WHERE', ']', '{', '|','}', '[', '(', ')', '<', '>', '#', '@', '!', '$', '%', '^', '+', '=', '?', '/', '\\', '`', '~', ';', ',', '.', '\'', '\"', '\n', '\t', '\r']	
                for char in invalid_chars:
                    pred = pred.replace(char, "")
                
                # Collapse multiple spaces into one
                pred = " ".join(pred.split())

                results.append((subj, pred, obj))
            return results