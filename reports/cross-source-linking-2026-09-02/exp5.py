import sqlite3
import numpy as np
import sqlite_vec
from assimilator.embeddings import embed_text

A = "Office of the Director of National Intelligence (ODNI) Unidentified Anomalous Phenomena (UAP) luring operation (2025-12)"
B = "Department of Defense UAP Intercept Operation at White Sands (November 2025-August 2026)"
B2 = "Department of Defense (DoD) Unidentified Anomalous Phenomena (UAP) intercept operation at White Sands (2025-11 to 2026-08)"
con = sqlite3.connect("/scratch/db2/knowledge.db")
con.enable_load_extension(True)
sqlite_vec.load(con)
names = {
    r[0]: (r[1], r[2])
    for r in con.execute("select id,node_type,name from nodes where retired_at is null")
}
ids = []
V = []
for nid, emb in con.execute("select node_id, embedding from vec_nodes"):
    if nid in names:
        ids.append(nid)
        V.append(np.frombuffer(emb, dtype=np.float32))
V = np.stack(V)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9


def vec(t):
    v = np.array(embed_text(t), dtype=np.float32)
    return v / np.linalg.norm(v)


va, vb, vb2 = vec(A), vec(B), vec(B2)
print(f"sim(A,B)={float(va @ vb):.3f}  sim(A,B normalised form)={float(va @ vb2):.3f}")
for tag, v in (("A (ODNI luring)", va), ("B (DoD intercept)", vb)):
    s = V @ v
    order = np.argsort(-s)
    print(f"\n{tag}: nearest existing nodes (project/event only, top 8)")
    k = 0
    for j in order:
        t, n = names[ids[j]]
        if t in ("project", "event"):
            print(f"   {s[j]:.3f} {t:8s} {n[:90]}")
            k += 1
        if k >= 8:
            break
    print(
        f"   rank of the other operation among ALL {len(ids)} nodes: {int((s > float(va @ vb)).sum()) + 1}"
    )
