# Discriminative power of IR metrics on NanoNQ: for each metric, how many model pairs differ significantly (paired sign-flip, 20k)
# and the median |t| = |mean(d)| / (sd(d)/sqrt(N)) across pairs. Old qrels: 9 models x 100 deep runs; relabel ("both"): 5 models, top-10 only.
import json, glob, math, itertools, os
import numpy as np
W = os.path.dirname(os.path.abspath(__file__)); S = f"{W}/data"
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
old = {}
for l in open(f"{S}/qrels.tsv"):
    q, d = l.split()[:2]; old.setdefault(q, set()).add(d)
A, B = {}, {}
for p, D in [("A", A), ("B", B)]:
    for f in glob.glob(f"{W}/{p}_labels*.jsonl"):
        for r in rd(f): D[(r["qid"], r["did"])] = int(r["label"])
both = {}
for k in A:
    if A[k] and B.get(k): both.setdefault(k[0], set()).add(k[1])
Q = sorted(old)
def metrics(run, rel, deep):
    ids = [d for d, _ in run]; g = [1 if d in rel else 0 for d in ids]
    first = next((i for i, x in enumerate(g) if x), None)
    dcg = lambda v: sum(x / math.log2(i + 2) for i, x in enumerate(v))
    out = {"hit@10": float(any(g[:10])), "MRR@10": 1 / (first + 1) if first is not None and first < 10 else 0.0,
           "P@10": sum(g[:10]) / 10, "nDCG@10": dcg(g[:10]) / dcg([1] * min(len(rel), 10)) if rel else 0.0}
    if deep:
        hits = 0; ap = 0
        for i, x in enumerate(g[:100]):
            if x: hits += 1; ap += hits / (i + 1)
        out.update({"nDCG@100": dcg(g[:100]) / dcg([1] * min(len(rel), 100)), "R@100": sum(g[:100]) / len(rel),
                    "AP@100": ap / len(rel), "MRR@100": 1 / (first + 1) if first is not None else 0.0})
    return out
def power(models, rel, deep):
    runs = {m: json.load(open(f"{S}/runs/{m}.json")) for m in models}
    sc = {m: [metrics(runs[m][q], rel[q], deep) for q in Q] for m in models}
    names = list(sc[models[0]][0]); rng = np.random.default_rng(0); res = {}
    for k in names:
        sig = 0; ts = []
        for x, y in itertools.combinations(models, 2):
            d = np.array([a[k] - b[k] for a, b in zip(sc[x], sc[y])])
            if not d.any(): ts.append(0); continue
            m = abs(d.mean()); t = m / (d.std(ddof=1) / math.sqrt(len(d))); ts.append(t)
            p = (np.abs((rng.choice([-1, 1], (20000, len(d))) * d).mean(1)) >= m - 1e-12).mean(); sig += p < .05
        res[k] = dict(sig=int(sig), pairs=len(ts), median_t=round(float(np.median(ts)), 2))
    return res
M9 = ["bm25", "e5-small", "e5-base", "e5-large", "lateon", "arctic-s", "bge-base", "mpnet-base", "qwen3-embed-0.6b", "static-mrl"]
out = {"old_10models_depth100": power(M9, old, True),
       "both_5models_top10": power(["bm25", "e5-small", "e5-base", "e5-large", "lateon"], {q: both.get(q, set()) | set() for q in Q}, False)}
for k, v in out.items():
    print(k)
    for m, r in sorted(v.items(), key=lambda x: -x[1]["median_t"]): print(f"  {m:9s} significant {r['sig']}/{r['pairs']}  median|t| {r['median_t']}")
json.dump(out, open(f"{W}/metrics_power.json", "w"), indent=1)
