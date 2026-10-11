# Binary relabel of NanoNQ top-10 (5 models): judge agreement, then ranks and paired intervals per label set.
import json, glob, os, math, itertools
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
S = f"{W}/data"
M = ["bm25", "e5-small", "e5-base", "e5-large", "lateon"]
old = {}
for l in open(f"{S}/qrels.tsv"):
    q, d = l.split()[:2]; old.setdefault(q, {})[d] = 1
pool = json.load(open(f"{W}/pool5.json"))
keys = [(r["qid"], r["did"]) for r in pool]
gold = {(r["qid"], r["did"]) for r in pool if r["gold"]}

def load(p):
    out = {}
    for f in sorted(glob.glob(f"{W}/{p}_labels*.jsonl")):
        for l in open(f):
            if l.strip(): r = json.loads(l); out[(r["qid"], r["did"])] = int(r["label"])
    return out
A, B, C = load("A"), load("B"), load("C")
assert all(k in A and k in B for k in keys), "missing A/B labels"

def kappa(x, y):
    x, y = np.array(x), np.array(y); po = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return po, (po - pe) / (1 - pe)
a = [A[k] for k in keys]; b = [B[k] for k in keys]
po, k = kappa(a, b)
dis = [k_ for k_ in keys if A[k_] != B[k_]]
print(f"pairs {len(keys)} | A positive {sum(a)} | B positive {sum(b)} | agreement {po:.3f} | Cohen kappa {k:.3f} | disagreements {len(dis)}")
print(f"gold ({len(gold)}): A says 1 for {sum(A[g] for g in gold)}, B for {sum(B[g] for g in gold)}")
json.dump([dict(qid=q, did=d) for q, d in dis], open(f"{W}/disagreements.json", "w"))

final = {}
for k_ in keys:
    final[k_] = A[k_] if A[k_] == B[k_] else C.get(k_)
missing = sum(v is None for v in final.values())
print("final labels missing (need pass C):", missing)

def qrels(lbl, add_gold):
    qr = {}
    for (q, d), v in lbl.items():
        if v == 1: qr.setdefault(q, {})[d] = 1
    if add_gold:
        for q, v in old.items():
            for d in v: qr.setdefault(q, {})[d] = 1
    return qr

runs = {m: json.load(open(f"{S}/runs/{m}.json")) for m in M}
Q = sorted(old)
dcg = lambda g: sum(x / math.log2(i + 2) for i, x in enumerate(g))
def nd(r, rel):
    i = dcg(sorted(rel.values(), reverse=True)[:10]); return dcg([rel.get(d, 0) for d, _ in r[:10]]) / i if i else 0.0

sets = [("old", old), ("A", qrels(A, False)), ("B", qrels(B, False))]
if not missing: sets.append(("final", qrels(final, False)))
sets.append(("both", qrels({k_: A[k_] & B[k_] for k_ in keys}, False)))
sets.append(("either", qrels({k_: A[k_] | B[k_] for k_ in keys}, False)))
out = {"judges": dict(pairs=len(keys), agreement=round(po, 4), kappa=round(k, 4), disagreements=len(dis),
                      gold=len(gold), gold_A=sum(A[g] for g in gold), gold_B=sum(B[g] for g in gold))}
for name, qr in sets:
    rel = np.mean([len(qr.get(q, {})) for q in Q]); empty = sum(not qr.get(q) for q in Q)
    sc = {m: np.array([nd(runs[m][q], qr.get(q, {})) for q in Q]) for m in M}
    print(f"\n== {name}: rel/query {rel:.2f}, queries without relevant {empty}")
    print("   " + "  ".join(f"{m} {100*sc[m].mean():.1f}" for m in sorted(M, key=lambda m: -sc[m].mean())))
    rng = np.random.default_rng(0); rows = []
    for x, y in itertools.combinations(M, 2):
        d = sc[x] - sc[y]
        if d.mean() < 0: x, y, d = y, x, -d
        m = d.mean(); h = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
        p = (np.abs((rng.choice([-1, 1], (20000, len(d))) * d).mean(1)) >= m - 1e-12).mean()
        rows.append(dict(a=x, b=y, diff=round(100 * m, 1), hw=round(100 * h, 1), p=round(p, 3), wins=int((d > 0).sum()), losses=int((d < 0).sum())))
        print(f"   {x:9s}>{y:9s} {100*m:5.1f} ±{100*h:4.1f} p={p:.3f} {rows[-1]['wins']}:{rows[-1]['losses']}")
    out[name] = dict(rel_per_query=round(rel, 2), empty_queries=empty, scores={m: round(100 * sc[m].mean(), 1) for m in M}, pairs=rows)
def tau(x, y):
    c = d = 0
    for i, j in itertools.combinations(M, 2):
        s_ = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += s_ > 0; d += s_ < 0
    return (c - d) / (len(M) * (len(M) - 1) / 2)
names = [n for n in out if n != "judges"]
out["kendall_tau"] = {f"{a}-{b}": round(tau(out[a]["scores"], out[b]["scores"]), 3) for a, b in itertools.combinations(names, 2)}
print("\nKendall tau of model rankings:", out["kendall_tau"])
json.dump(out, open(f"{W}/results.json", "w"), indent=1)
