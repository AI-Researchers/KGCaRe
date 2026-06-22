prompt_template = [
    {"role": "system", "content": (
        "You are a ConditionalQA assistant. Answer only from the provided evidence passages. "
        "The question begins with a marker like [Question Type: yes/no_conditional]. "
        "Return only valid JSON with keys `Thought`, `Answer`, and `Conditions`. "
        "If there are no applicable conditions, set `Conditions` to an empty list."
    )},
    {"role": "user", "content": (
        "Wikipedia Title: Benefit Rules\n"
        "People can keep Housing Benefit if they reached State Pension age before the cut-off date.\n"
        "Wikipedia Title: Exceptions\n"
        "If a claimant is under State Pension age and has no protected status, they need to claim Universal Credit instead.\n\n"
        "Question: [Question Type: yes/no_conditional] Can I keep Housing Benefit?\n"
    )},
    {"role": "assistant", "content": (
        '{"Thought": "The passage says benefit depends on reaching State Pension age and otherwise the claimant must claim Universal Credit.", "Answer": "yes", "Conditions": ["Reached State Pension age before the cut-off date"]}'
    )},
    {"role": "user", "content": "${prompt_user}"},
]
