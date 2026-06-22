from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType




MLT_PROMPT_1 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to extract "
    "and construct a well-structured, comprehensive, and contextually accurate "
    "knowledge graph in the form of (subject, predicate, object) triplets from the provided text. "
    "Extract every relevant subject-predicate-object triple in such a way that re-verbalizing these triples preserves all key information from the original text without any data loss. "
    "This information will be later used to answer multi-hop questions.\n\n"

    "### Step 1: Analyze and Understand the Text ###\n"
    "1. Carefully read the text to identify factual statements and key assertions.\n"
    "2. Understand how different facts connect across multiple paragraphs.\n"
    "3. Capture core facts, definitions, relationships, and reasoning chains relevant to typical multi-hop questions.\n\n"

    "Example for Context Understanding (HotpotQA):\n"
    "Question: 'What is the religion of the author of *The Selfish Gene*?'\n"
    "Input Text: 'Richard Dawkins wrote *The Selfish Gene*. Richard Dawkins is an atheist.'\n"
    "Context Analysis: The text connects the author to the book, and then provides the religion of the person.\n\n"

    "### Step 2: Extract Key Entities ###\n"
    "1. Identify specific named entities, people, works, events, concepts, etc.\n"
    "2. Avoid vague references; make implicit subjects explicit based on context.\n\n"
    "Example for Entity Extraction:\n"
    "- Subject: 'Richard Dawkins'\n"
    "- Predicate: 'wrote'\n"
    "- Object: 'The Selfish Gene'\n\n"

    "### Step 3: Extract Relationships ###\n"
    "1. Capture facts, causal links, and indirect reasoning steps.\n"
    "2. Include entity-attribute-value triples and bridge facts connecting different pieces of knowledge.\n\n"
    "Example for Relationship Extraction:\n"
    "- ('Richard Dawkins', 'wrote', 'The Selfish Gene')\n"
    "- ('Richard Dawkins', 'is', 'atheist')\n"
    "- ('The Selfish Gene', 'was written by', 'Richard Dawkins')\n\n"

    "------------------------\n"
    "{text}\n"
    "------------------------\n"
    "### Output Format ###\n"
    "Context Analysis:\n"
    "- Summary of how facts relate and how they could help answer a question\n\n"
    "Entities:\n"
    "- List of key entities\n\n"
    "Relationships:\n"
    "- (Entity 1, Predicate, Entity 2)\n"
    "- (Entity 3, Predicate, Entity 4)\n"
)


MLT_PROMPT_2 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to use the output of a previous analysis "
    "and additional unstructured text to extract bridge facts, entity reasoning chains, and formulate initial knowledge graph triplets. "
    "These will support answering complex multi-hop questions from HotpotQA.\n\n"

    "### Inputs Provided ###\n"
    "1. **Extracted Entities and Relationships:**\n"
    "{intermediate_extracted}\n\n"
    "2. **Unstructured Text:**\n"
    "{text}\n\n"

    "### Step 4: Identify Bridge Facts and Reasoning Chains ###\n"
    "1. Identify reasoning steps connecting distant facts:\n"
    "   - e.g., 'Person A directed Movie X. Movie X starred Person B.' → helps connect Person A and Person B.\n"
    "2. Find implicit links such as co-location, temporal overlap, and factual dependencies.\n\n"

    "Example:\n"
    "Question: 'Which film starring Emma Stone was directed by Damien Chazelle?'\n"
    "Input:\n"
    "- Damien Chazelle directed *La La Land*.\n"
    "- Emma Stone starred in *La La Land*.\n"
    "Bridge Fact Chain:\n"
    "- Damien Chazelle → directed → La La Land → starred → Emma Stone\n\n"

    "### Step 5: Formulate Initial Triplets ###\n"
    "1. Integrate entity relationships and reasoning chains.\n"
    "2. Make sure each triplet is meaningful and contributes to multi-hop inference.\n\n"
    "### Output Format ###\n"
    "Bridge and Reasoning Chains:\n"
    "- Reasoning 1: A → B → C\n"
    "- Reasoning 2: X → Y → Z\n\n"
    "Initial Triplets:\n"
    "- (Subject 1, Predicate, Object)\n"
    "- (Subject 2, Predicate, Object)\n"
)


MLT_PROMPT_3 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to refine the provided triplets by enhancing "
    "logical connectivity and validating completeness and specificity using the original context and triplets. "
    "Ensure the final output supports multi-hop reasoning required to answer HotpotQA-style questions.\n\n"

    "### Inputs Provided ###\n"
    "1. **Initial Triplets:**\n"
    "{intermediate_extracted}\n\n"
    "2. **Unstructured Text:**\n"
    "{text}\n\n"

    "### Step 6: Enhance Logical Connectivity ###\n"
    "1. Merge and bridge related triplets into logical multi-hop chains.\n"
    "2. Use reasoning such as transitive or referential links where possible.\n\n"
    "Example:\n"
    "- ('Einstein', 'born in', 'Ulm')\n"
    "- ('Ulm', 'is in', 'Germany')\n"
    "- Add: ('Einstein', 'born in', 'Germany') ← inferred\n\n"

    "### Step 7: Validate Completeness and Specificity ###\n"
    "1. Ensure coverage of all facts needed for answering multi-hop questions.\n"
    "2. Avoid vague entries like ('He', 'did', 'that'); replace with explicit terms.\n\n"
    "### Output Format ###\n"
    "Return only the final triplets in the format (subject, predicate, object), one per line. "
    "Do not include explanations, extra text, or malformed triples.\n\n"

    "Enhanced Triplets:\n"
    "(Subject 1, Predicate, Object)\n"
    "(Subject 2, Predicate, Object)\n"
)
