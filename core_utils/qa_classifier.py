from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings


def classify_single_question(references):
    """
    Classify question type based on references.

    Args:
        references (List): Answer references in the format [[answer, conditions], ...]

    Returns:
        str: One of 'yes/no', 'yes/no_conditional', 'span', 'span_conditional', or 'unanswerable'
    """
    if not references:
        return "unanswerable"

    # Yes/No
    if any(ans[0] in ["yes", "no"] for ans in references):
        return "yes/no_conditional" if any(ans[1] for ans in references) else "yes/no"

    # Span
    return "span_conditional" if any(ans[1] for ans in references) else "span"


class QuestionTypeClassifier:
    """
    Classifies a question as either 'Yes/No' or 'Span' using ICL examples and the configured LLM.
    """

    def __init__(self):
        self.examples = [
            {"question": "Do I have a greater right to probate in respect of my late father's estate?", "type": "Yes/No"},
            {"question": "When can I start paying the inheritance tax?", "type": "Span"},
            {"question": "Is interest payable if I agree to pay it in installments?", "type": "Yes/No"},
            {"question": "Am I eligible for the Maternity allowance and if so how much will I get?", "type": "Span"},
            {"question": "Can I apply to change my father's will, or is this a matter for the deputy handling his financial affairs?", "type": "Yes/No"},
            {"question": "How long can I get help for after my main benefits stop?", "type": "Span"},
            {"question": "Do we have to pay the High Income Child Benefit Tax Charge?", "type": "Yes/No"},
            {"question": "What level of Blind Person's Allowance can I claim?", "type": "Span"},
        ]

        self.task_description = (
            "Your task is to classify whether a question should be answered with Yes/No or a full span. "
            "I'll give you some examples first. You should only answer 'Yes/No' or 'Span'."
        )

        self.prompt_template = "Question: {question}\nQuestion Type:"
        self.list_icl_chat_examples = [ChatMessage(role="system", content=self.task_description)]

        for ex in self.examples:
            self.list_icl_chat_examples.append(ChatMessage(role="user", content=self.prompt_template.format(question=ex["question"])))
            self.list_icl_chat_examples.append(ChatMessage(role="assistant", content=ex["type"]))

    def classify(self, question: str) -> str:
        """
        Classify the question type via LLM prompt.

        Args:
            question (str): The input question.

        Returns:
            str: 'Yes/No' or 'Span'
        """
        messages = self.list_icl_chat_examples.copy()
        messages.append(ChatMessage(role="user", content=self.prompt_template.format(question=question)))
        return Settings.llm.chat(messages).message.content
