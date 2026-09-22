import sqlite3
import math
import collections
import numpy as np
import sqlite_vec

LT = "9e6a5228-6f47-4442-9d77-c677b9cdf1af"
RC = "308dd5bd-eed8-4f08-b15a-98f7bb989522"
con = sqlite3.connect("/scratch/db/knowledge.db")
con.enable_load_extension(True)
sqlite_vec.load(con)
titles = {r[0]: r[1][:70] for r in con.execute("select id,title from records")}
claims = con.execute("select id, record_id, content from claims").fetchall()
cid = [c[0] for c in claims]
crec = np.array([c[1] for c in claims])
ctext = {c[0]: c[2] for c in claims}
idx = {c: i for i, c in enumerate(cid)}
V = np.zeros((len(cid), 1024), dtype=np.float32)
have = np.zeros(len(cid), bool)
for claim_id, emb in con.execute("select claim_id, embedding from vec_claims"):
    i = idx.get(claim_id)
    if i is not None:
        V[i] = np.frombuffer(emb, dtype=np.float32)
        have[i] = True
print("claims", len(cid), "embedded", have.sum())
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
recs = sorted(set(crec))
# --- B1 record centroids
C = {r: V[(crec == r) & have].mean(0) for r in recs}
for r in C:
    C[r] /= np.linalg.norm(C[r]) + 1e-9


def rank_centroid(r):
    sims = sorted(((float(C[r] @ C[o]), o) for o in recs if o != r), reverse=True)
    return sims


print("\n=== B1 record-centroid neighbours")
for a, b, name in ((LT, RC, "LT->RC"), (RC, LT, "RC->LT")):
    sims = rank_centroid(a)
    pos = [o for _, o in sims].index(b) + 1
    print(
        f"{name}: rank {pos} of {len(sims)}  sim={dict((o, s) for s, o in sims)[b]:.3f}"
    )
    for s, o in sims[:5]:
        print(f"   {s:.3f} {titles[o]}")


# --- B3-lite claim kNN aggregated by record
def knn_records(r, k=20):
    ii = np.where((crec == r) & have)[0]
    S = V[ii] @ V.T
    S[:, ii] = -1  # exclude own record
    agg = collections.defaultdict(list)
    for row in S:
        top = np.argpartition(-row, k)[:k]
        for j in top:
            agg[crec[j]].append(float(row[j]))
    scored = sorted(
        ((len(v), float(np.mean(v)), o) for o, v in agg.items()), reverse=True
    )
    return scored


print("\n=== B3 claim-kNN (k=20) hits aggregated by record")
for a, b, name in ((LT, RC, "LT->RC"), (RC, LT, "RC->LT")):
    sc = knn_records(a)
    order = [o for _, _, o in sc]
    pos = order.index(b) + 1 if b in order else None
    print(f"{name}: rank {pos} of {len(sc)}")
    for n, m, o in sc[:6]:
        print(f"   hits={n:3d} mean={m:.3f} {titles[o]}")
# --- B2 anchor-node restricted
print("\n=== B2 anchor-node: max claim-pair similarity between records sharing a node")
refs = con.execute(
    "select r.node_id, c.record_id, c.id from claim_node_refs r join claims c on c.id=r.claim_id"
).fetchall()
bynode = collections.defaultdict(lambda: collections.defaultdict(list))
for n, r, c in refs:
    bynode[n][r].append(idx[c])
nodename = {r[0]: r[1] for r in con.execute("select id,name from nodes")}
shared = [n for n in bynode if LT in bynode[n] and RC in bynode[n]]
for n in shared:
    groups = bynode[n]
    rs = list(groups)
    pairs = []
    for i, a in enumerate(rs):
        A = V[groups[a]]
        for b in rs[i + 1 :]:
            B = V[groups[b]]
            pairs.append((float((A @ B.T).max()), a, b))
    pairs.sort(reverse=True)
    pos = [k for k, (s, a, b) in enumerate(pairs) if {a, b} == {LT, RC}][0] + 1
    print(
        f"{nodename[n][:60]}: {len(rs)} records, LT-RC max-sim rank {pos} of {len(pairs)}; top: {pairs[0][0]:.3f}"
    )
# --- hub-weighted shared-node score
print("\n=== shared-node idf score, LT vs every record")
nrec = {n: len(bynode[n]) for n in bynode}
N = len(recs)
ltnodes = set(n for n in bynode if LT in bynode[n])
sc = []
for o in recs:
    if o == LT:
        continue
    s = sum(math.log(N / nrec[n]) for n in ltnodes if o in bynode[n])
    sc.append((s, o))
sc.sort(reverse=True)
pos = [o for _, o in sc].index(RC) + 1
print(f"RC rank {pos} of {len(sc)}")
for s, o in sc[:5]:
    print(f"   {s:.2f} {titles[o]}")
