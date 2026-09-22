import json
import subprocess
import sys
import time
import sqlite3

S = "/tmp/claude-1000/-home-mark-repos-anomalica-master/e057f75c-24f5-4f23-ad24-9c973052562d/scratchpad"
exec(open(f"{S}/judge.py").read().split("pairs=[")[0])
STRICT = """You are comparing the extracted claims of two source records from a reference corpus on anomalous phenomena.

Question: do the two records refer to the SAME SPECIFIC THING - one identifiable incident, operation, programme, document or investigation, pinned by a place, a date, an official identity, or distinctive described particulars that BOTH records carry?

Not a shared subject: the same speaker or outlet, the same period or administration, the same agency, the same general topic, or "the government's UAP efforts" in general. Two records by the same journalist are not about the same thing because he made both. Default to "unrelated" unless you can point to a claim on each side that pins the same specific thing.

- "same_subject": both records clearly refer to the same specific thing (possibly under different descriptions).
- "possibly_related": the particulars on each side (place, date, actors, described events) could be the same specific thing but neither record establishes it.
- "unrelated": everything else.

Give the shared specific thing as one short noun phrase with its date or place (empty if unrelated), a two-sentence reason that quotes the pinning particular from each side, and up to 6 claim-id pairs that carry the connection.

RECORD A:
{A}

RECORD B:
{B}
"""
con = sqlite3.connect(f"{S}/db/knowledge.db")


def rid(pat):
    return con.execute(
        "select id from records where title like ?", (f"%{pat}%",)
    ).fetchone()[0]


def claims_of(r):
    rows = con.execute(
        "select id, claim_type, content from claims where record_id=? order by location_in_record",
        (r,),
    ).fetchall()
    return "\n".join(f"[{r[0][:8]}] ({r[1]}) {r[2].strip()}" for r in rows)


pairs = [
    ("Luring Operation", "Skywatcher didn"),
    ("Luring Operation", "David Grusch: The Whistleblower"),
    ("Luring Operation", "USPER Narrative"),
    ("Luring Operation", "Episode 73"),
    ("Skywatcher didn", "7NEWS"),
    ("Skywatcher didn", "Answers Your Biggest"),
    ("Skywatcher didn", "Dr. Phil read"),
    ("Skywatcher didn", "David Grusch: The Whistleblower"),
    ("Luring Operation", "Pressure Mounts"),
]
out = []
for a, b in pairs:
    prompt = STRICT.format(A=claims_of(rid(a)), B=claims_of(rid(b)))
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
        json.dumps(schema),  # noqa: F821 - loaded by the experiment bootstrap above
        "--output-format",
        "json",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
    try:
        wrap, _ = json.JSONDecoder().raw_decode(p.stdout.strip())
        res = (
            wrap.get("structured_output")
            or json.JSONDecoder().raw_decode(wrap.get("result", "{}"))[0]
        )
    except Exception as e:
        res = {"error": str(e), "stdout": p.stdout[:300]}
    res.update(pair=f"{a} vs {b}", seconds=round(time.time() - t0))
    res.pop("links", None)
    out.append(res)
    print(json.dumps(res))
    sys.stdout.flush()
    json.dump(out, open(f"{S}/judge-results3.json", "w"), indent=1)
