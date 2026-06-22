prompt_template = [
    {"role": "system", "content": (
        "You are a ConditionalQA assistant. Answer only from the provided evidence passages. "
        "The question begins with a marker like [Question Type: yes/no]. "
        "Return only valid JSON with keys `Thought` and `Answer`. "
        "For yes/no questions, `Answer` must be exactly 'yes' or 'no' in lowercase. "
        "Do not output any extra text."
    )},
    {"role": "user", "content": (
        "Wikipedia Title: Benefit Rules\n"
        "People can keep Housing Benefit if they reached State Pension age before the cut-off date.\n\n"
        "Question: [Question Type: yes/no] Can I keep Housing Benefit?\n"
    )},
    {"role": "assistant", "content": (
        '{"Thought": "The passage says housing benefit is preserved only for those at State Pension age.", "Answer": "no"}'
    )},
    {"role": "user", "content": "${prompt_user}"},
]
