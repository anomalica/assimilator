import sqlite3
import numpy as np
import sqlite_vec
import collections
import json

LT = "9e6a5228-6f47-4442-9d77-c677b9cdf1af"
RC = "308dd5bd-eed8-4f08-b15a-98f7bb989522"
con = sqlite3.connect("/scratch/db/knowledge.db")
con.enable_load_extension(True)
sqlite_vec.load(con)
titles = {r[0]: r[1] for r in con.execute("select id,title from records")}
ncl = {
    r[0]: r[1] for r in con.execute("select record_id,count(*) from claims group by 1")
}
claims = con.execute("select id, record_id from claims").fetchall()
cid = [c[0] for c in claims]
crec = np.array([c[1] for c in claims])
idx = {c: i for i, c in enumerate(cid)}
V = np.zeros((len(cid), 1024), dtype=np.float32)
for claim_id, emb in con.execute("select claim_id, embedding from vec_claims"):
    i = idx.get(claim_id)
    if i is not None:
        V[i] = np.frombuffer(emb, dtype=np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9


def shortlist(r, k=20, top=15):
    ii = np.where(crec == r)[0]
    S = V[ii] @ V.T
    S[:, ii] = -1
    agg = collections.Counter()
    for row in S:
        for j in np.argpartition(-row, k)[:k]:
            agg[crec[j]] += 1
    return [(o, n) for o, n in agg.most_common(top)]


out = {}
for tag, r in (("LT", LT), ("RC", RC)):
    out[tag] = [
        {"id": o, "hits": n, "claims": ncl[o], "title": titles[o][:80]}
        for o, n in shortlist(r)
    ]
    print(tag)
    [
        print(f"   hits={e['hits']:3d} claims={e['claims']:5d} {e['title']}")
        for e in out[tag]
    ]
json.dump(out, open("/scratch/shortlist.json", "w"), indent=1)
