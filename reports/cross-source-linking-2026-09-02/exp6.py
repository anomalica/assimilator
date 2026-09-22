import sqlite3
import yaml
import re
from assimilator.entity_reranker import Entity, entity_from_graph, get_reranker

lt = yaml.safe_load(open("/scratch/lt2.yaml"))


def claims_for(d, needle):
    return [
        c["text"].strip()
        for c in d["domain_claims"]
        if any(needle in r["name"] for r in c.get("refs", []))
    ]


a_claims = (
    claims_for(lt, "luring operation (December 2025)")
    + [
        c["text"].strip()
        for c in lt["domain_claims"]
        if re.search(r"lur", c["text"], re.I)
    ][:5]
)
A = Entity(
    name="Office of the Director of National Intelligence (ODNI) Unidentified Anomalous Phenomena (UAP) luring operation (2025-12)",
    node_type="project",
    claims=a_claims[:6],
)
rc = (
    yaml.safe_load(
        open(
            "/home/nonroot/workspace/../digests/2026-08-09-video-ross-coulthart-q-a-skywatcher-didn-t-disappear-it-went.yaml"
        )
    )
    if False
    else None
)

rcp = "/scratch/rc_canonical.yaml"
rc = yaml.safe_load(open(rcp))
b_claims = [
    c["text"].strip()
    for c in rc["domain_claims"]
    if re.search(r"\boperation|intercept|summon", c["text"], re.I)
][:6]
B = Entity(
    name="Department of Defense (DoD) Unidentified Anomalous Phenomena (UAP) intercept operation at White Sands (2025-11 to 2026-08)",
    node_type="project",
    claims=b_claims,
)
con = sqlite3.connect("/scratch/db2/knowledge.db")


def node(pat):
    r = con.execute(
        "select id,name from nodes where retired_at is null and name like ? order by length(name) limit 1",
        (f"%{pat}%",),
    ).fetchone()
    return entity_from_graph(con, r[0]) if r else None


D = {
    k: node(p)
    for k, p in {
        "UAPTF": "Task Force (UAPTF)",
        "AAWSAP": "(AAWSAP)",
        "ODNI prelim": "Office of the Director of National Intelligence Unidentified Aerial Phenomena (UAP) prelim",
        "Late 2025 test-range encounter": "Late 2025 Unidentified Aerial Phenomena (UAP) encounter at a U.S. weapons test range",
        "Sierra Blanca TFR": "Sierra Blanca Temporary Flight Restriction",
    }.items()
}
pairs = (
    [("A vs B (the two operations)", A, B)]
    + [(f"A vs {k}", A, e) for k, e in D.items() if e]
    + [(f"B vs {k}", B, e) for k, e in D.items() if e]
)
rr = get_reranker()
scores = rr.score([(a, b) for _, a, b in pairs])
for (label, _, _), s in zip(pairs, scores):
    print(f"{s:.3f}  {label}")
