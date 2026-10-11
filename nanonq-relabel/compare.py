# Is model A really better than B on your own benchmark?
# Input: per-query metric of both models (same queries, same order), e.g. nDCG@10.
# Output: mean difference with a 95% bootstrap interval, wins/losses, and how many queries
# you need to reliably detect a given gain (paired test, power 80%, p<0.05): N ≈ (2.8·σ/Δ)².
import numpy as np

def compare(metric_a, metric_b, delta=0.05, seed=0):
    d = np.asarray(metric_a, float) - np.asarray(metric_b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (10_000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    print(f"A - B: {d.mean():+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]"
          f"{'  <- includes zero: no proven gain' if lo <= 0 <= hi else ''}")
    print(f"queries: A wins {(d > 0).sum()}, B wins {(d < 0).sum()}, ties {(d == 0).sum()}")
    sd = d.std(ddof=1)
    print(f"spread of the per-query difference: {sd:.3f}; to detect a gain of {delta}: ~{round((2.8 * sd / delta) ** 2)} queries")

if __name__ == "__main__":
    # demo on this folder's data: LateOn vs e5-large after relabel
    import json, glob, math, os
    W = os.path.dirname(os.path.abspath(__file__))
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    A = {(r["qid"], r["did"]): r["label"] for f in glob.glob(f"{W}/A_labels*.jsonl") for r in rd(f)}
    B = {(r["qid"], r["did"]): r["label"] for f in glob.glob(f"{W}/B_labels*.jsonl") for r in rd(f)}
    rel = {}
    for k, v in A.items():
        if v and B.get(k): rel.setdefault(k[0], set()).add(k[1])
    Q = sorted({q for q, _ in A})
    dcg = lambda g: sum(x / math.log2(i + 2) for i, x in enumerate(g))
    nd = lambda run, r: dcg([d in r for d, _ in run[:10]]) / dcg([1] * min(len(r), 10)) if r else 0.0
    score = lambda m: [nd(json.load(open(f"{W}/data/runs/{m}.json"))[q], rel.get(q, set())) for q in Q]
    compare(score("lateon"), score("e5-large"))
