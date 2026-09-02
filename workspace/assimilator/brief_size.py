"""How big a brief is, so a consumer can choose a model that holds it.

The brief used to be capped at 600 claims. That number was set against model
context limits that no longer exist - every model in the policy file now carries
a window of a million tokens or more, except Haiku - and it was cutting evidence
off exactly the subjects a reader is most likely to look up: 600 of 2,457 claims
for Whitley Strieber, so 76% of what the graph knows about him could not appear
in his article or be cited from it, with nothing downstream able to tell it
existed.

Two things replace it. The brief is sized in TOKENS against the window of the
stage that consumes it, and if it still has to be cut it says so.

WHAT IS SIZED is the claim material a consumer renders - content and excerpt,
plus a line of framing per claim - and not the YAML file. The file carries
about 1,190 characters of ids, hashes, slugs and provenance around every claim,
nearly four times the claim text, and none of it reaches a model. The largest
brief is 3.6 MB on disk and renders to roughly 286,000 tokens; sized as a file
it would read as over a window it fits in with room to spare, and the cut it
would trigger would drop evidence for nothing.
"""

from __future__ import annotations

# Measured over 12 real briefs with cl100k: mean 2.80 characters per token on
# the YAML, range 2.71 to 3.10; the assembler fitted 2.6 on its rendered prose
# against real usage on the Claude path. NOT the usual 4 - claim text is dense
# with names, dates and quotation, and assuming 4 understates a large brief's
# real size by about 30%.
#
# The conservative end of the measured range is used deliberately. Fewer
# characters per token means MORE estimated tokens, so the error runs towards
# picking a model that is larger than needed rather than one that cannot hold
# the input. Overflowing a context window is a silent failure at the provider;
# choosing a bigger model costs money.
CHARS_PER_TOKEN = 2.7

# Share of the window left for the consumer's own prompt and its output. The
# brief is the bulk of the input but not all of it.
PROMPT_AND_OUTPUT_HEADROOM = 0.15


def estimate_tokens(text: str) -> int:
    """Token estimate for a string. An ESTIMATE - see CHARS_PER_TOKEN."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def budget_for(window: int) -> int:
    """Tokens a brief may occupy given the consuming stage's context window."""
    if window <= 0:
        return 0
    return int(window * (1 - PROMPT_AND_OUTPUT_HEADROOM))


def consuming_window(stage: str = "assemble") -> int:
    """The context window a brief must fit, from the policy file (ADR 0047).

    Returns 0 if the policy cannot be read, and the caller must then NOT cap:
    silently falling back to a small number would reintroduce the fault this
    module exists to remove, and would do it invisibly.
    """
    try:
        from anomalica_common import model_policy as mp

        return mp.load().stage_context_window(stage)
    except Exception:  # pragma: no cover - policy unreadable
        return 0
