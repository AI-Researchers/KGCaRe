prompt_template = [
    {"role": "system", "content": (
        "You are a ConditionalQA assistant. Answer only from the provided evidence passages. "
        "The question begins with a marker like [Question Type: span_conditional]. "
        "Return only valid JSON with keys `Thought`, `Answer`, and `Conditions`. "
        "If there are no applicable conditions, set `Conditions` to an empty list."
    )},
    {"role": "user", "content": (
        "Wikipedia Title: Benefit Rules\n"
        "People can keep Housing Benefit if they reached State Pension age before the cut-off date.\n"
        "Wikipedia Title: Exceptions\n"
        "If a claimant is under State Pension age and has no protected status, they need to claim Universal Credit instead.\n\n"
        "Question: [Question Type: span_conditional] What is the condition for keeping Housing Benefit?\n"
    )},
    {"role": "assistant", "content": (
        '{"Thought": "The passage says the claimant must have reached State Pension age before the cut-off date.", "Answer": "Reached State Pension age before the cut-off date", "Conditions": ["Reached State Pension age before the cut-off date"]}'
    )},
    {"role": "user", "content": "${prompt_user}"},
]
