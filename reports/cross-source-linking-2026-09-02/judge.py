import json
import subprocess
import sys
import time

S = "/tmp/claude-1000/-home-mark-repos-anomalica-master/e057f75c-24f5-4f23-ad24-9c973052562d/scratchpad"


def load(tag):
    return json.load(open(f"{S}/claims_{tag}.json"))


def fmt(tag):
    return "\n".join(
        f"[{tag}-{c['id']}] ({c['type']}) {c['text'].strip()}" for c in load(tag)
    )


schema = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["same_subject", "possibly_related", "unrelated"],
        },
        "shared_subject": {"type": "string"},
        "reason": {"type": "string"},
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["same_fact", "same_subject", "contradicts"],
                    },
                },
                "required": ["a", "b", "relation"],
            },
        },
    },
    "required": ["verdict", "shared_subject", "reason", "links"],
}
PROMPT = """You are comparing the extracted claims of two source records from a reference corpus on anomalous phenomena. Each record already shares generic entities with dozens of others (agencies, the topic "UAP"), so shared generic entities do NOT count.

Question: do the two records discuss the SAME SPECIFIC real-world subject - one particular operation, programme, incident, investigation or effort - that is more specific than a standing agency or the general topic? Answer:
- "same_subject" if both records clearly describe the same specific operation/incident/programme (even under different descriptions).
- "possibly_related" if they plausibly describe the same specific thing or directly connected things, but neither record establishes it (different places/dates/actors could be the same, could not).
- "unrelated" otherwise, including when they merely share the general topic, an agency, or a period.

Name the shared specific subject in one short noun phrase (empty if unrelated). Give the reason in two sentences. List the claim pairs that carry the connection (ids as given, at most 8 pairs), relation "same_fact" when both assert the same fact, "same_subject" when they are about the same specific thing, "contradicts" when they conflict.

RECORD A ({ta}):
{A}

RECORD B ({tb}):
{B}
"""
pairs = [
    ("LT", "RC"),
    ("LT", "USPER"),
    ("LT", "PRESSURE"),
    ("RC", "DEBRIEFED"),
    ("LT", "WATCHDOG"),
]
titles = {
    "LT": "web article, 2026-07-12",
    "RC": "video transcript, 2026-08-09",
    "USPER": "government document",
    "PRESSURE": "web article",
    "DEBRIEFED": "video transcript",
    "WATCHDOG": "web article",
}
out = []
for a, b in pairs:
    prompt = PROMPT.format(ta=titles[a], tb=titles[b], A=fmt(a), B=fmt(b))
    cmd = [
        "claude",
        "-p",
        "--model",
        "claude-haiku-4-5",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
        "--tools",
        "",
        "--effort",
        "low",
        "--json-schema",
        json.dumps(schema),
        "--output-format",
        "json",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
    try:
        wrap = json.loads(p.stdout)
        res = wrap.get("structured_output") or json.loads(wrap.get("result", "{}"))
    except Exception as e:
        res = {"error": str(e), "stdout": p.stdout[:500], "stderr": p.stderr[:500]}
    res["pair"] = f"{a}-{b}"
    res["seconds"] = round(time.time() - t0)
    res["usage"] = wrap.get("usage") if isinstance(wrap, dict) else None
    out.append(res)
    print(json.dumps({k: v for k, v in res.items() if k != "usage"}, indent=1))
    sys.stdout.flush()
json.dump(out, open(f"{S}/judge-results.json", "w"), indent=1)
