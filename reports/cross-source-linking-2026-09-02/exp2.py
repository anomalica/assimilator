import sqlite3
import re
import numpy as np
import sqlite_vec

LT = "9e6a5228-6f47-4442-9d77-c677b9cdf1af"
RC = "308dd5bd-eed8-4f08-b15a-98f7bb989522"
con = sqlite3.connect("/scratch/db/knowledge.db")
con.enable_load_extension(True)
sqlite_vec.load(con)
titles = {r[0]: r[1][:45] for r in con.execute("select id,title from records")}
claims = con.execute("select id, record_id, content from claims").fetchall()
cid = [c[0] for c in claims]
crec = np.array([c[1] for c in claims])
ctext = [c[2] for c in claims]
idx = {c: i for i, c in enumerate(cid)}
V = np.zeros((len(cid), 1024), dtype=np.float32)
for claim_id, emb in con.execute("select claim_id, embedding from vec_claims"):
    i = idx.get(claim_id)
    if i is not None:
        V[i] = np.frombuffer(emb, dtype=np.float32)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
pat = re.compile(r"lur|summon|shoot|intercept|attract", re.I)
for r, other, name in ((LT, RC, "LT"), (RC, LT, "RC")):
    ii = [i for i in np.where(crec == r)[0] if pat.search(ctext[i])]
    print(f"\n=== {name}: {len(ii)} theme claims; top-5 cross-record neighbours each")
    for i in ii:
        s = V[i] @ V.T
        s[crec == r] = -1
        top = np.argsort(-s)[:5]
        ranks_other = np.where(crec[np.argsort(-s)] == other)[0]
        print(f"- {ctext[i][:110]}")
        print(
            f"    first neighbour from the other record: rank {ranks_other[0] + 1 if len(ranks_other) else None}, sim {s[np.argsort(-s)][ranks_other[0]]:.3f}"
            if len(ranks_other)
            else "    none"
        )
        for j in top:
            print(f"    {s[j]:.3f} [{titles[crec[j]]}] {ctext[j][:90]}")
