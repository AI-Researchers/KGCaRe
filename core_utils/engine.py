import logging
from typing import Optional
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from core_utils import all_prompts as cs_prompts


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class RAGQueryEngine(CustomQueryEngine):
    """Custom Query Engine for both KG and Vector retrieval."""

    retriever: Optional[BaseRetriever] = None
    response_synthesizer: BaseSynthesizer

    def custom_query(self,  query_str: str, q_type: str, index_type: str = "hybrid_index", data_FS = None) :
        """
        Args:
            data_FS: few-shot examples
            query_str (str): Full question
            q_type (str): Question type ("yes/no", "span", etc.)
            index_type (str): Retrieval type ("vector_index", "kg_index", "hybrid_index", "no_index")
        """
        logger.info(f"Question Type: {q_type}")

        # Select prompt based on type
        if q_type == "yes/no":
            qa_prompt = cs_prompts.yes_no_qa_s4_prompt
        elif q_type == "yes/no_conditional":
            qa_prompt = cs_prompts.yes_no_con_qa_s4_prompt
        elif q_type == "span":
            qa_prompt = cs_prompts.span_qa_s4_prompt
        elif q_type == "span_conditional":
            qa_prompt = cs_prompts.span_con_qa_s4_prompt
        else:
            qa_prompt = cs_prompts.span_qa_s4_prompt

        query_bundle = QueryBundle(query_str)

        # Retrieve context based on index type
        if index_type == "vector_index":
            nodes = self.retriever._retrieve(query_bundle)
            context_str = "\n\n".join([n.node.get_content() for n in nodes])

        elif index_type == "kg_index":
            nodes = self.retriever._retrieve(query_bundle)
            context_str = "\n\n Knowledge Triples: " + "\n\n".join([n.node.get_content() for n in nodes])

        elif index_type == "hybrid_index":
            _, dict_retrieve_nodes = self.retriever._retrieve(query_bundle)
            text_context = "\n\n".join([n.node.get_content() for n in dict_retrieve_nodes["vector"]])
            kg_context = "\n\n".join([n.node.get_content() for n in dict_retrieve_nodes["kg"]])
            context_str = text_context + "\n\n Knowledge Triples: " + kg_context

        elif index_type == "no_index":
            context_str = "\n\n NO CONTEXT AVAILABLE: \n Answer the question based on your knowledge. \n\n"

        else:
            raise ValueError(f"Unknown index type: {index_type}")
    

        # Create pseudo-node for synthesizer
        context_nodes = [NodeWithScore(node=TextNode(text=context_str))]
        print(context_str)
        ####---------WITH RANDOM SHOTS-------------####
        # Update the prompt with few shots examples
        # few_shot_examples = select_random_few_shots(data_FS, q_type, args.num_shots)
        
        # prompt_ = qa_prompt.partial_format(few_shot_examples=few_shot_examples,context_str=context_str)
        # prompt_out = qa_prompt.format(few_shot_examples=few_shot_examples,query_str=query_str,context_str=context_str)
        ####---------WITH STATIC SHOTS-------------####

        # Static few-shot (hardcoded) — optional: make dynamic later
        prompt_ = qa_prompt.partial_format(context_str=context_str)
        prompt_out = qa_prompt.format(query_str=query_str, context_str=context_str)

        # Update response synthesizer and get response
        self.response_synthesizer.update_prompts({"text_qa_template": prompt_})
        response_obj = self.response_synthesizer.synthesize(query_bundle, context_nodes)

        return response_obj, prompt_out
