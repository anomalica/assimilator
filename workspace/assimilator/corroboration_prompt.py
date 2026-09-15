"""Canonical prompt used to adjudicate candidate corroboration pairs."""

CORROBORATION_VERIFY_PROMPT = """Below are pairs of claims from different records. For each pair, decide whether they assert the SAME underlying fact or are genuinely DIFFERENT assertions.

RULES:
- "same": the claims make the same factual assertion, possibly with different wording or detail level.
- "different": the claims are about different things, even if they are thematically related.
- Two claims about the same TOPIC but making different ASSERTIONS are "different".
  Example: "The object was 12 metres long" and "The object had no wings" are both about the object, but different assertions.
- Two claims making the same ASSERTION in different words are "same".
  Example: "The object traversed 100km in seconds" and "The UAP covered approximately 100 kilometres almost instantly" are the same.

{pairs_text}

OUTPUT FORMAT (respond with ONLY valid JSON, no markdown fencing):

{{"decisions": [
    {{"pair_id": 1, "verdict": "same"}},
    {{"pair_id": 2, "verdict": "different"}}
]}}"""
