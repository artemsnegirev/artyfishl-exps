# How fragile is LateOn > e5-large after relabel ("both" judges): wins/losses, how many queries to drop to flip the leader,
# random half-splits that disagree, and scores of all dense models without the 2 queries most favouring LateOn (-> drop2.json).
import json, glob, math, os
import numpy as np
W = os.path.dirname(os.path.abspath(__file__)); S = f"{W}/data"
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
A, B = {}, {}
for p, D in [("A", A), ("B", B)]:
    for f in glob.glob(f"{W}/{p}_labels*.jsonl"):
        for r in rd(f): D[(r["qid"], r["did"])] = int(r["label"])
rel = {}
for k in A:
    if A[k] and B.get(k): rel.setdefault(k[0], set()).add(k[1])
Q = sorted({q for q, _ in A})
dcg = lambda g: sum(x / math.log2(i + 2) for i, x in enumerate(g))
nd = lambda run, r: dcg([1 if d in r else 0 for d, _ in run[:10]]) / dcg([1] * min(len(r), 10)) if r else 0.0
M = ["lateon", "e5-large", "e5-small", "e5-base"]
sc = {m: np.array([nd(json.load(open(f"{S}/runs/{m}.json"))[q], rel.get(q, set())) for q in Q]) for m in M}
d = sc["lateon"] - sc["e5-large"]
print(f"LateOn - e5-large: {100*d.mean():+.1f}, wins {(d>0).sum()}, losses {(d<0).sum()}, ties {(d==0).sum()}, sd {100*d.std(ddof=1):.1f}")
s = np.sort(d)[::-1]; k, cur = 0, d.sum()
while cur > 0: cur -= s[k]; k += 1
print("queries to drop to flip the leader:", k)
rng = np.random.default_rng(0); flips = 0
for _ in range(20000):
    p = rng.permutation(len(Q)); flips += np.sign(d[p[:25]].mean()) != np.sign(d[p[25:]].mean())
print("random 25/25 splits naming different winners:", flips / 20000)
drop = np.argsort(-d)[:2]; keep = np.setdiff1d(np.arange(len(Q)), drop)
out = dict(dropped=[Q[i] for i in drop], scores={m: round(float(100 * sc[m][keep].mean()), 1) for m in M})
print("without", out["dropped"], out["scores"])
json.dump(out, open(f"{W}/drop2.json", "w"))
