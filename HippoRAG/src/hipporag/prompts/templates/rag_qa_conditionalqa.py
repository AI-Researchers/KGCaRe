rag_qa_system = (
    "You are a ConditionalQA assistant. Answer only from the provided evidence passages. "
    "The user question begins with a marker like [Question Type: yes/no], [Question Type: yes/no_conditional], [Question Type: span], or [Question Type: span_conditional]. "
    "Return only valid JSON with keys `Answer` and, when applicable, `Conditions`. "
    "Do not add any extra explanation outside the JSON object. "
    "For yes/no questions, `Answer` must be exactly 'yes' or 'no'."
)

generic_example = (
    "Wikipedia Title: Benefit Rules\n"
    "People can keep Housing Benefit if they reached State Pension age before the cut-off date.\n"
    "Wikipedia Title: Exceptions\n"
    "If a claimant is under State Pension age and has no protected status, they need to claim Universal Credit instead.\n\n"
)

prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": (
        f"{generic_example}"
        "Question: [Question Type: yes/no] Can I keep Housing Benefit?\n"
    )},
    {"role": "assistant", "content": (
        '{"Thought": "The passage says housing benefit is preserved only for those at State Pension age.", "Answer": "no"}'
    )},
    {"role": "user", "content": "${prompt_user}"},
]
