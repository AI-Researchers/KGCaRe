prompt_template = [
    {"role": "system", "content": (
        "You are a ConditionalQA assistant. Answer only from the provided evidence passages. "
        "The question begins with a marker like [Question Type: span]. "
        "Return only valid JSON with keys `Thought` and `Answer`. "
        "Do not output any extra text."
    )},
    {"role": "user", "content": (
        "Wikipedia Title: Benefit Rules\n"
        "People can keep Housing Benefit if they reached State Pension age before the cut-off date.\n\n"
        "Question: [Question Type: span] What is required to keep Housing Benefit?\n"
    )},
    {"role": "assistant", "content": (
        '{"Thought": "The passage says the claimant must have reached State Pension age before the cut-off date.", "Answer": "Reached State Pension age before the cut-off date"}'
    )},
    {"role": "user", "content": "${prompt_user}"},
]
