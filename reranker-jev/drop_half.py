"""Pessimistic check of the relabelling: drop a random half of the positives the judge added (pairs MIRACL never
saw) and recompute nDCG@10, mean over 20 seeds: drop_half.json.

    .venv/Scripts/python reranker-jev/drop_half.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402
from evaluate import ARMS, NAMES, load_scores, metrics, order  # noqa: E402

SEEDS, SHARE = 20, 0.5


def ndcg(items, runs, q, arm):
    return 100 * np.mean([metrics([q[x["qid"]][k] for k in order(runs[arm]["scores"][x["qid"]])],
                                  sum(q[x["qid"]]))["nDCG@10"] for x in items])


if __name__ == "__main__":
    fq = json.load(open(ROOT / "labels" / "final_qrels.json", encoding="utf-8"))["qrels"]
    items = [x for x in data.load() if x["qid"] in fq]
    runs = load_scores("miracl")
    base = {x["qid"]: [int(fq[x["qid"]].get(d["did"], d["rel"])) for d in x["docs"]] for x in items}
    added = [(x["qid"], k) for x in items for k, d in enumerate(x["docs"])
             if d["judged"] is None and base[x["qid"]][k]]
    per_seed = []
    for seed in range(SEEDS):
        rng = random.Random(seed)
        q = {k: list(v) for k, v in base.items()}
        for qid, k in added:
            if rng.random() < SHARE:
                q[qid][k] = 0
        per_seed.append({a: ndcg(items, runs, q, a) for a in ARMS})
    out = {"queries": len(items), "added_positives": len(added), "share_dropped": SHARE, "seeds": SEEDS,
           "relabelled": {a: round(ndcg(items, runs, base, a), 1) for a in ARMS},
           "drop_half": {a: round(float(np.mean([m[a] for m in per_seed])), 1) for a in ARMS}}
    json.dump(out, open(ROOT / "drop_half.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for a in sorted(ARMS, key=lambda a: -out["drop_half"][a]):
        print(f"  {NAMES[a]:38s} relabelled {out['relabelled'][a]:5.1f}  drop half {out['drop_half'][a]:5.1f}")
