"""Prompt constants used by the refactored KGCaRe pipeline.

The KG construction prompts below are copied from the current implementation and
must not be semantically rewritten. They define the paper's 3-step KG creation
process.
"""

from __future__ import annotations

from typing import Literal

DatasetName = Literal["conditionalqa", "hotpotqa"]


CONDITIONALQA_MLT_PROMPT_1 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to extract "
    "and construct a well-structured, comprehensive, and contextually accurate "
    "knowledge graph in the form of (subject, predicate, object) triplets from the provided text. "
    "Extract every relevant subject-predicate-object triple from the text in such a way that re-verbalizing these triples preserves all key information from the original text without any data loss."
    "Follow the systematic steps below to ensure thoroughness, logical connectivity, and context-specific accuracy:\n\n"
    "### Step 1: Analyze and Understand the Text ###\n"
    "1. Carefully read the provided text to understand its overall context, purpose, and primary actions.\n"
    "2. Determine the scope of the text, including:\n"
    "   - Specific conditions, rules, or exceptions.\n"
    "   - Procedural steps or alternative actions.\n"
    "3. Take note of any temporal, causal, or conditional relationships, including:\n"
    "   - Deadlines or sequences of events.\n"
    "   - Dependencies or consequences of actions.\n"
    "   - Edge cases or exceptions.\n\n"
    "Example for Context Understanding:\n"
    "Input Text: 'Applicants must submit the form within 30 days of receiving the decision.'\n"
    "Context Analysis: This text provides a procedural rule with a temporal condition "
    "('within 30 days') and an action ('submit the form').\n\n"
    "### Step 2: Extract Key Entities ###\n"
    "1. Identify meaningful entities, including subjects, objects, and key concepts, such as:\n"
    "   - Individuals: e.g., 'Applicant', 'Guardian'.\n"
    "   - Organizations: e.g., 'Court of Protection', 'HMRC'.\n"
    "   - Documents: e.g., 'Visa application', 'Grant proposal'.\n"
    "   - Concepts or actions: e.g., 'VAT exemption', 'Joint decision-making'.\n"
    "2. Exclude generic references (e.g., 'you', 'it') and replace them with specific terms reflecting context.\n"
    "   - Generic: 'You' → Contextual: 'Visa applicant'.\n"
    "   - Generic: 'It' → Contextual: 'The document', 'The application'.\n\n"
    "Example for Entity Extraction:\n"
    "Input Text: 'Applicants must submit the form within 30 days of receiving the decision.'\n"
    "Entities:\n"
    "- Subject: 'Applicant'\n"
    "- Predicate: 'submit'\n"
    "- Object: 'form'\n\n"
    "### Step 3: Extract Relationships ###\n"
    "1. Identify relationships connecting the extracted entities. Focus on:\n"
    "   - Explicit actions: e.g., 'requires', 'prohibits', 'allows'.\n"
    "   - Procedural links: e.g., 'must be approved by', 'needs to be filed with'.\n"
    "   - Conditional relations: e.g., 'if consent is not obtained, seek court approval'.\n"
    "2. Capture implied relationships, such as:\n"
    "   - Temporal dependencies: e.g., 'must be submitted within 12 months'.\n"
    "   - Causal or procedural links: e.g., 'failure to comply leads to rejection'.\n\n"
    "Example for Relationship Extraction:\n"
    "Input Text: 'Applicants must submit the form within 30 days of receiving the decision.'\n"
    "Relationships:\n"
    "- ('Applicant', 'must submit', 'form')\n"
    "- ('form', 'must be submitted within', '30 days')\n"
    "- ('30 days', 'depends on', 'receiving the decision')\n\n"
    "------------------------\n"
    "{text}\n"
    "------------------------\n"
    "### Output Format ###\n"
    "Context Analysis:\n"
    "- Summary of the context and purpose of the text\n\n"
    "Entities:\n"
    "- List of extracted entities\n\n"
    "Relationships:\n"
    "- (Entity 1, Predicate, Entity 2)\n"
    "- (Entity 3, Predicate, Entity 4)\n"
)


CONDITIONALQA_MLT_PROMPT_2 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to use the output of a previous analysis "
    "and additional unstructured text to extract conditional and alternative scenarios, and to formulate initial knowledge graph triplets. "
    "Extract every relevant subject-predicate-object triple from the text in such a way that re-verbalizing these triples preserves all key information from the original text without any data loss."
    "Ensure that the output is logically connected and contextually accurate.\n\n"

    "### Inputs Provided ###\n"
    "1. **Extracted Entities and Relationships:**\n"
    "{intermediate_extracted}\n\n"
    "2. **Unstructured Text:**\n"
    "{text}\n\n"

    "### Instructions ###\n"
    "Follow these steps to complete the task:\n\n"

    "### Step 4: Address Conditional and Alternative Scenarios ###\n"
    "1. Identify all conditional or alternative pathways explicitly or implicitly stated in the text:\n"
    "   - For example: 'If consent is not given, then permission must be obtained from the court.'\n"
    "   - For example: 'Zero-rated sales are excluded from VAT registration threshold calculations.'\n"
    "2. Capture nuances of exceptions, alternative actions, or scenarios where specific conditions apply.\n\n"

    "### Step 5: Formulate Initial Triplets ###\n"
    "1. Combine the extracted entities and relationships with the identified conditional and alternative scenarios.\n"
    "2. Formulate clear (subject, predicate, object) triplets that capture:\n"
    "   - Meaningful actions and rules.\n"
    "   - Conditions and their consequences.\n"
    "   - Alternative pathways or exceptions.\n"
    "3. Ensure each triplet is distinct and logically connected.\n"
    "4. Avoid redundant or vague triplets.\n\n"

    "### Output Format ###\n"
    "Conditional and Alternative Scenarios:\n"
    "- Condition 1: If X, then Y\n"
    "- Condition 2: Alternative action A if action B is not possible\n\n"

    "Initial Triplets:\n"
    "- (Subject 1, Predicate, Object)\n"
    "- (Subject 2, Predicate, Object)\n"
    "- (Subject 3, Predicate, Object)\n"
)


CONDITIONALQA_MLT_PROMPT_3 = (
    "You are an expert in Knowledge Graph (KG) construction. Your task is to refine the provided triplets by enhancing "
    "logical connectivity and validating completeness and specificity. Use the output of the previous analysis and the original text "
    "Extract every relevant subject-predicate-object triple from the text in such a way that re-verbalizing these triples preserves all key information from the original text without any data loss."
    "to ensure the knowledge graph is thorough, accurate, and logically connected.\n\n"

    "### Inputs Provided ###\n"
    "1. **Initial Triplets:**\n"
    "{intermediate_extracted}\n\n"
    "2. **Unstructured Text:**\n"
    "{text}\n\n"

    "### Instructions ###\n"
    "Follow these steps systematically to refine and enhance the knowledge graph:\n\n"

    "### Step 6: Enhance Logical Connectivity ###\n"
    "1. Interconnect triplets to reflect complex relationships:\n"
    "   - Use the object of one triplet as the subject of another if logically connected.\n"
    "   - For example:\n"
    "     - ('Guardian', 'needs consent from', 'Parents').\n"
    "     - ('Parents', 'can delegate consent to', 'Court of Protection').\n"
    "     - Derived Triplet: ('Guardian', 'can delegate consent to', 'Court of Protection').\n"
    "2. Add new triplets to represent implicit relationships or logical extensions:\n"
    "   - For example:\n"
    "     - If ('A', 'requires', 'B') and ('B', 'enables', 'C'), create ('A', 'enables', 'C').\n\n"

    "### Step 7: Validate Completeness and Specificity ###\n"
    "1. Review the triplets to ensure they cover all explicit and implicit details from the text.\n"
    "2. Validate that the triplets address:\n"
    "   - Procedural or conditional scenarios.\n"
    "   - Exceptions or edge cases.\n"
    "3. Ensure that all relationships are specific and unambiguous:\n"
    "   - Avoid generic or vague relationships (e.g., 'Gift', 'has duty', 'Customs').\n"
    "   - Prefer specific relationships (e.g., 'Gift under £100', 'is exempt from', 'Customs duty').\n\n"

    "### Output Format ###\n"
    "Return only the final triplets in the strict format (subject, predicate, object), one triplet per line. "
    "Ensure the triplets are free of extraneous text, redundant characters, or missing components.\n\n"

    "Enhanced Triplets:\n"
    "(Subject 1, Predicate, Object)\n"
    "(Subject 2, Predicate, Object)\n"
    "(Subject 3, Predicate, Object)\n"
)


HOTPOTQA_MLT_PROMPT_1 = (
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


HOTPOTQA_MLT_PROMPT_2 = (
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


HOTPOTQA_MLT_PROMPT_3 = (
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


KEYWORD_EXTRACT_PROMPT = (
    "A question is provided below. Given the question, extract up to {max_keywords} keywords from the text. "
    "Focus on extracting the keywords that we can use to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
)


PRUNE_TRIPLE_PROMPT = """Perform following tasks:
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


REASONING_PROMPT = """Perform following tasks:
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


REASONING_WITHOUT_KG_PROMPT = """Given a question, identify specific, concise entities essential for answering the question and enclose each in curly brackets {xxxx}. Keep entities as short as possible, focusing on core concepts.
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


def kg_prompt_chain(dataset: DatasetName) -> tuple[str, str, str]:
    if dataset == "hotpotqa":
        return HOTPOTQA_MLT_PROMPT_1, HOTPOTQA_MLT_PROMPT_2, HOTPOTQA_MLT_PROMPT_3
    return CONDITIONALQA_MLT_PROMPT_1, CONDITIONALQA_MLT_PROMPT_2, CONDITIONALQA_MLT_PROMPT_3


QA_SYSTEM_PROMPT = (
    "You are a careful RAG question answering assistant. Answer using only the provided context. "
    "Return structured JSON matching the requested schema. For ConditionalQA, include conditions only when "
    "the answer depends on specific provided context statements."
)


CONDITIONALQA_YESNO_QA_USER_PROMPT = """Dataset: ConditionalQA
Question type: {qtype}

You are a helpful assistant that answers questions using only the provided Context information and Knowledge Triples, not prior knowledge.
Answers can be yes or no. You have to answer yes or no and nothing else in the answer field. Do not answer "it depends" or anything similar.
You HAVE to choose only yes or no, even if you are uncertain.

Some same-type examples are given below.
---------------------
Knowledge Triples: (finance plan, available for, Green Deal assessment improvements)
(loan, available for, insulation replacement)
Question: I own a home and energy bills are always high. When I hired a engineer to assess why my energy bills are high. He mentioned the problem is because of insulation. I can't afford to replace the insulation. These energy improvements have been recommended in my Green Deal assessment. Can I get a loan to replace the insulation ?
Answer: yes

Knowledge Triples: (employer, applies to, Central Arbitration Committee)
(trade union, derecognised if, employed less than 21 people for 13 weeks)
Question: I am the owner of a high street retailing business which has fallen on hard times in the wake of the pandemic. At our peak we had over 150 employees, who were represented by a trade union whom we voluntarily recognised six years ago. Now our headcount has fallen below 20, and the remaining workers include a union shop steward who is being unreasonably obstructive of our attempts to turn the company around. I have had only 17 employees for the last year. Given the shrunken state of the business, is it now possible to get the union derecognised?
Answer: yes

Knowledge Triples: (test room, no access to, personal items)
(mobile phones, stored in, locker or plastic box)
Question: My friend Ann has booked her ADI part 1 test. She has scheduled an important phone call a few minutes before the exam day. Will she be allowed to access her mobile phone during exam day?
Answer: no

Knowledge Triples: (caregiver, receives, GBP 67.60 a week)
(caregiver, cares for, someone at least 35 hours a week)
(Carer's Allowance, eligibility requires, 35 hours of care)
(partner, receives, Disability Living Allowance at higher rate)
Question: I live in Scotland, and currently care for my physically disabled partner for approximately 30 hrs per week. My partner receives Disability Living Allowance at the higher rate and I work part time, earning approximately GBP 500pcm. Can I claim Carer's Allowance to help with my partner's care?
Answer: no
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Return only JSON with fields answer_type, answers, and rationale.
Set answer_type to "yes_no". answers must contain exactly one object with answer exactly "yes" or "no" and conditions as an empty list."""


CONDITIONALQA_SPAN_QA_USER_PROMPT = """Dataset: ConditionalQA
Question type: {qtype}

You are a helpful assistant that answers questions using only the provided Context information and Knowledge Triples, not prior knowledge.
Answers must be a short span extracted from the provided evidence. Do not answer "it depends" or anything similar.
You have to extract only the answer span. Do not add explanation in the answer field.
The context may include knowledge triples in the format (subject, predicate, object). Use these triples as part of the Context information to identify relevant information for answering the question.

Some same-type examples are given below.
---------------------
Knowledge Triples: (judge, agrees, court sends certificate)
(court, sends, certificate)
(certificate, takes, several weeks)
Question: Me and my wife were married but with the mutual consent we applied for divorce. We have applied for a decree nisi and are now waiting for the result What happens to my application and how I will be communicated with?
Answer: if the judge agrees, the court will send you and your husband or wife a certificate. this may take several weeks.

Knowledge Triples: (apply, requires, age 18 or over)
Question: I am a 16 year old living in Derby and was born male, I do not feel like I fit into any gender category. What age can I apply for a certificate?
Answer: 18 or over

Knowledge Triples: (first step, download and fill in, notice of appeal form)
Question: I am a 21 year old single Algerian male who is claiming asylum on the grounds of persecution. I applied for some support whilst I get myself established here but just received a letter saying I've been declined. What is the first step in the appeals process?
Answer: download and fill in a notice of appeal form

Knowledge Triples: (EWC, requires, at least 1,000 employees in EEA)
(EWC, requires, 150 employees in each of at least 2 countries in EEA)
Question: I work for a firm with several offices across Europe. I have heard of EWC but I am not sure if we qualify. Before I apply, what do we need to do to qualify?
Answer: at least 1,000 employees in the eea
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Return only JSON with fields answer_type, answers, and rationale.
Set answer_type to "span". answers must contain exactly one object with a concise answer span and conditions as an empty list."""


CONDITIONALQA_YESNO_CONDITIONAL_QA_USER_PROMPT = """Dataset: ConditionalQA
Question type: {qtype}

You are a helpful assistant that answers questions using only the provided Context information and Knowledge Triples, not prior knowledge.
Answers can be yes or no. You have to answer yes or no and nothing else in the answer field. Do not answer "it depends" or anything similar.
You HAVE to choose only yes or no, even if you are uncertain.
Some answers may require the assumption of some sentences or triples from the evidence to be true. If that is the case, write the full supporting condition statements in the conditions list. You must have used those conditions in your reasoning.

Some same-type examples are given below.
---------------------
Knowledge Triples: (arrange, payment plan, HMRC)
(payment plan, requires, owe GBP 30,000 or less)
(payment plan, requires, no other payment plans or debts with HMRC)
(payment plan, requires, tax returns up to date)
(payment plan, requires, less than 60 days after payment deadline)
Question: I am self employed as a caterer and always do my own taxes. I was informed recently that I had made a mistake on my last self assessment form and owe quite a substantial amount of money to HMRC. I cannot pay at the present time. Is there anything I can do to delay the repayments I have to make?
Answer: yes Conditions: owe GBP 30,000 or less; no other payment plans or debts with HMRC; tax returns are up to date; less than 60 days after the payment deadline

Knowledge Triples: (Parents' Learning Allowance, available for, full-time student with children)
(claim, available for, full-time undergraduate course)
(claim, available for, Initial Teacher Training course)
Question: I have two children and I am currently due to start a degree course in social work this coming Autumn. Can I make a claim given that I will be studying and no longer working?
Answer: yes Conditions: full-time undergraduate course; Initial Teacher Training course

Knowledge Triples: (passport, requires, blank page for visa)
(applicant, from, outside the EU/Switzerland/Norway/Iceland/Liechtenstein)
(applicant, from, EU/Switzerland/Norway/Iceland/Liechtenstein without biometric passport)
Question: I am 26 and planning on applying for a start-up visa for my new business in the UK. I speak fluent English. I have a valid passport but all the pages are full. Can I use my current passport for my application?
Answer: no Conditions: from outside the EU, Switzerland, Norway, Iceland or Liechtenstein; from the EU, Switzerland, Norway, Iceland or Liechtenstein but do not have a biometric passport with a chip in it

Knowledge Triples: (home, must be offered to, old landlord or another social landlord)
(sell, restricted in, national park)
(sell, restricted in, area of outstanding natural beauty)
(sell, restricted in, rural area for Right to Buy)
Question: I purchased my current family home in Norwich, England from the local council 9 years ago under the right-to-buy scheme. With my children having now grown up and left home my partner and I are now looking to sell the house and retire to something a little smaller. Can I now sell my home to whomever I like on the open market?
Answer: no Conditions: national park; area of outstanding natural beauty; area the government says is rural for Right to Buy
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Return only JSON with fields answer_type, answers, and rationale.
Set answer_type to "yes_no". answers must contain exactly one object with answer exactly "yes" or "no". Put required assumptions or supporting condition statements in conditions. Use an empty list only if no condition is needed."""


CONDITIONALQA_SPAN_CONDITIONAL_QA_USER_PROMPT = """Dataset: ConditionalQA
Question type: {qtype}

You are a helpful assistant that answers questions using only the provided Context information and Knowledge Triples, not prior knowledge.
Answers must be a short span extracted from the provided evidence. Do not answer "it depends" or anything similar.
You have to extract only the answer span. Do not add explanation in the answer field.
Some answers may require the assumption of some sentences or triples from the evidence to be true. If that is the case, write the full supporting condition statements in the conditions list. You must have used those conditions in your reasoning.
The context may include knowledge triples in the format (subject, predicate, object). Use these triples as part of the Context information to identify relevant information for answering the question.

Some same-type examples are given below.
---------------------
Knowledge Triples: (gowns and PPE, classified as, healthcare offensive waste)
(offensive waste, defined as, non-clinical waste)
(offensive waste, not contain, pharmaceutical or chemical substances)
(offensive waste, may be unpleasant to, anyone who comes into contact)
(healthcare offensive waste, includes, outer dressings and protective clothing like masks, gowns, gloves)
(healthcare offensive waste, non-hazardous, 18-01-04 | 18-02-03)
Question: I am a GP partner in a small, private medical practice in London. We have recently gotten through a lot of gowns and other PPE as a result of the COVID-19 pandemic, and these now need to be classified for disposal. How are potentially COVID-infected gowns and other PPE to be classified?
Answer: healthcare offensive waste Conditions: offensive waste is non-clinical waste; offensive waste does not contain pharmaceutical or chemical substances; healthcare offensive waste includes outer dressings and protective clothing like masks, gowns and gloves

Knowledge Triples: (ask for, permission to appeal to the Upper Tribunal)
(appeal to Upper Tribunal, if, legal mistake with tribunal's decision)
(tribunal, got, law wrong)
(tribunal, did not apply, correct law)
(tribunal, did not follow, correct procedures)
(tribunal, had no, evidence to support its decision)
Question: I have recently had my appeal rejected and I am at a loss of what to do my family members are all in the uk and I wont be able to see them anymore. Is there anything I can do as my appeal has been rejected?
Answer: you can ask for permission to appeal to the upper tribunal Conditions: legal mistake with the tribunal's decision; tribunal got the law wrong; tribunal did not apply the correct law; tribunal did not follow the correct procedures; tribunal had no evidence to support its decision

Knowledge Triples: (eligible for, Maternity Allowance)
(Maternity Allowance, amount, GBP 151.97 a week or 90% of average weekly earnings)
(Maternity Allowance, paid if, cannot get Statutory Maternity Pay)
(Maternity Allowance, requires, employed or self-employed for at least 26 weeks)
(Maternity Allowance, requires, earning GBP 30 a week or more in at least 13 weeks)
Question: I am twenty weeks pregnant and work part time at present. I am not currently in receipt of any other benefits. Am I eligible for the Maternity allowance and if so how much will I get?
Answer: GBP 151.97 a week or 90% of your average weekly earnings Conditions: cannot get Statutory Maternity Pay; employed or self-employed for at least 26 weeks; earning GBP 30 a week or more in at least 13 weeks

Knowledge Triples: (use, DBS Adult First)
(DBS Adult First, available if, provide care services for adults)
(care services, example, care home)
Question: I want to recruit an employee urgently to work in a care home.I want to know her DBS checks in the next 2 days. Which service can i use?
Answer: dbs adult first Conditions: provide care services for adults
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Return only JSON with fields answer_type, answers, and rationale.
Set answer_type to "span". answers must contain exactly one object with a concise answer span. Put required assumptions or supporting condition statements in conditions. Use an empty list only if no condition is needed."""


HOTPOTQA_SPAN_QA_USER_PROMPT = """Dataset: HotPotQA
Question type: {qtype}

You are a HotPotQA assistant that answers using only the provided Context information and Knowledge Triples.
The examples below demonstrate the required output shape for span-answer questions.
---------------------
Knowledge Triples:
(La La Land, directed by, Damien Chazelle)
(La La Land, starred, Emma Stone)
Question: Which film starring Emma Stone was directed by Damien Chazelle?
Answer JSON: {{"answer_type":"span","answers":[{{"answer":"La La Land","conditions":[]}}],"rationale":"The triples connect Emma Stone and Damien Chazelle through La La Land."}}

Knowledge Triples:
(The Selfish Gene, written by, Richard Dawkins)
(Richard Dawkins, religion, atheist)
Question: What is the religion of the author of The Selfish Gene?
Answer JSON: {{"answer_type":"span","answers":[{{"answer":"atheist","conditions":[]}}],"rationale":"The author is Richard Dawkins, and the triples identify him as atheist."}}

Knowledge Triples:
(The Shining, written by, Stephen King)
(Stephen King, born in, Portland, Maine)
Question: Which city is the birth place of the author of The Shining?
Answer JSON: {{"answer_type":"span","answers":[{{"answer":"Portland, Maine","conditions":[]}}],"rationale":"The author is Stephen King, and his birthplace is Portland, Maine."}}

Knowledge Triples:
(Frodo Baggins, portrayed by, Elijah Wood)
(Elijah Wood, born in, Cedar Rapids)
Question: Who portrayed Frodo Baggins?
Answer JSON: {{"answer_type":"span","answers":[{{"answer":"Elijah Wood","conditions":[]}}],"rationale":"The triples state that Frodo Baggins was portrayed by Elijah Wood."}}
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Answer rules:
- This is a span-answer HotPotQA question.
- Do not answer yes or no.
- Set answer_type to "span" when the context supports a short answer.
- Extract the shortest entity, name, date, number, place, title, or phrase that answers the question.
- Use one concise answer in answers[0].answer.
- If the provided context and knowledge triples do not support an answer, set answer_type to "unanswerable" and return an empty answers list.
- Return JSON with fields answer_type, answers, and rationale."""


HOTPOTQA_YESNO_QA_USER_PROMPT = """Dataset: HotPotQA
Question type: {qtype}

You are a HotPotQA assistant that answers using only the provided Context information and Knowledge Triples.
The examples below demonstrate the required output shape for yes/no questions.
---------------------
Knowledge Triples:
(Laleli Mosque, located in, Laleli)
(Esma Sultan Mansion, located in, Ortakoy)
Question: Are Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
Answer JSON: {{"answer_type":"yes_no","answers":[{{"answer":"no","conditions":[]}}],"rationale":"The two places are located in different neighborhoods."}}

Knowledge Triples:
(Scott Derrickson, nationality, American)
(Ed Wood, nationality, American)
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Answer JSON: {{"answer_type":"yes_no","answers":[{{"answer":"yes","conditions":[]}}],"rationale":"Both people are identified as American."}}

Knowledge Triples:
(Jane Eyre, written by, Charlotte Bronte)
(Wuthering Heights, written by, Emily Bronte)
Question: Were Jane Eyre and Wuthering Heights written by the same author?
Answer JSON: {{"answer_type":"yes_no","answers":[{{"answer":"no","conditions":[]}}],"rationale":"The works have different authors."}}

Knowledge Triples:
(La La Land, directed by, Damien Chazelle)
(Damien Chazelle, born in, Providence, Rhode Island)
(Providence, Rhode Island, located in, United States)
Question: Was the director of La La Land born in the United States?
Answer JSON: {{"answer_type":"yes_no","answers":[{{"answer":"yes","conditions":[]}}],"rationale":"The director is Damien Chazelle, who was born in Providence, Rhode Island, in the United States."}}
---------------------

Context information:
{vector_context}

Knowledge Triples:
{kg_context}

Question:
{question}

Answer rules:
- This is a yes/no HotPotQA question.
- Set answer_type to "yes_no".
- Use one concise answer in answers[0].answer.
- The answer must be exactly "yes" or "no".
- If the provided context and knowledge triples do not support a yes/no decision, set answer_type to "unanswerable" and return an empty answers list.
- Return JSON with fields answer_type, answers, and rationale."""


def hotpotqa_qa_user_prompt_template(qtype: str) -> str:
    normalized = qtype.strip().lower().replace("_", "/")
    if normalized in {"yes/no", "yes-no", "yesno"}:
        return HOTPOTQA_YESNO_QA_USER_PROMPT
    return HOTPOTQA_SPAN_QA_USER_PROMPT


def conditionalqa_qa_user_prompt_template(qtype: str) -> str:
    normalized = qtype.strip().lower().replace("_", "/")
    if normalized in {"yes/no/conditional", "yes/no-conditional"}:
        return CONDITIONALQA_YESNO_CONDITIONAL_QA_USER_PROMPT
    if normalized in {"span/conditional", "span-conditional"}:
        return CONDITIONALQA_SPAN_CONDITIONAL_QA_USER_PROMPT
    if normalized in {"yes/no", "yes-no", "yesno"}:
        return CONDITIONALQA_YESNO_QA_USER_PROMPT
    return CONDITIONALQA_SPAN_QA_USER_PROMPT


def build_qa_user_prompt(
    *,
    dataset: DatasetName,
    question: str,
    vector_context: str,
    kg_context: str,
    qtype: str = "span",
) -> str:
    if dataset == "conditionalqa":
        template = conditionalqa_qa_user_prompt_template(qtype)
    elif dataset == "hotpotqa":
        template = hotpotqa_qa_user_prompt_template(qtype)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return template.format(
        qtype=qtype,
        vector_context=vector_context,
        kg_context=kg_context,
        question=question,
    )
