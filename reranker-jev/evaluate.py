"""Metrics and pairwise significance from saved per-query scores; nothing is rerun.

    .venv/Scripts/python reranker-jev/evaluate.py      -> results.json

Ranking: score descending; ties keep the (seeded, shuffled) data order, the same for every arm.
nDCG@10 (binary gains, ideal over all relevant candidates in the pool) and MRR@10, x100.
Significance: paired sign-flip randomisation test on per-query nDCG@10 (and MRR@10), 10k permutations,
two-sided, as in nanonq-relabel/discriminative.py.
qrels: "original" = MTEB MIRACLReranking labels; "relabelled" = relabel.py's final labels (if present).
"""
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

ARMS = ["jev-pw", "jev-list", "fd", "bge", "qwen3-0.6b", "kalm-nano-r2"]
NAMES = {"jev-pw": "Jev pointwise (API)", "jev-list": "Jev listwise (API)", "fd": "FRIDA-Decisions",
         "bge": "bge-reranker-v2-m3", "qwen3-0.6b": "Qwen3-Reranker-0.6B", "kalm-nano-r2": "KaLM-Reranker-V1-Nano-R2"}
PERMS = 10_000


def order(scores):
    return sorted(range(len(scores)), key=lambda k: -scores[k])   # stable: ties keep data order


def metrics(ranked_rel, n_rel):
    g = ranked_rel[:10]
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, 10)))
    first = next((i for i, x in enumerate(g) if x), None)
    return {"nDCG@10": dcg / idcg if idcg else 0.0, "MRR@10": 1 / (first + 1) if first is not None else 0.0}


def load_scores(name):
    out = {}
    for arm in ARMS:
        p = ROOT / "scores" / name / f"{arm}.json"
        if p.exists():
            out[arm] = json.load(open(p, encoding="utf-8"))
    return out


def per_query(items, runs, qrels):
    """{arm: {metric: np.array over items}}"""
    res = {}
    for arm, r in runs.items():
        rows = []
        for x in items:
            rel = qrels[x["qid"]]
            ranked = [rel[k] for k in order(r["scores"][x["qid"]])]
            rows.append(metrics(ranked, sum(rel)))
        res[arm] = {m: np.array([row[m] for row in rows]) for m in rows[0]}
    return res


def signflip(d, rng):
    if not d.any():
        return 1.0
    m = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], (PERMS, len(d)))
    return float((np.abs((flips * d).mean(1)) >= m - 1e-12).mean())


def table(pq, metric="nDCG@10"):
    arms = sorted(pq, key=lambda a: -pq[a][metric].mean())
    rng = np.random.default_rng(0)
    pairs = {}
    for a, b in itertools.combinations(arms, 2):
        d = pq[a][metric] - pq[b][metric]
        pairs[f"{a} vs {b}"] = {"diff": round(100 * d.mean(), 2), "wins": int((d > 0).sum()),
                                "losses": int((d < 0).sum()), "p": signflip(d, rng)}
    return arms, pairs


def evaluate_set(name, items, runs, qrels_by_name):
    out = {}
    for qname, qrels in qrels_by_name.items():
        pq = per_query(items, runs, qrels)
        arms, pairs = table(pq, "nDCG@10")
        _, pairs_mrr = table(pq, "MRR@10")
        out[qname] = {
            "relevant_per_query": round(statistics.mean(sum(v) for v in qrels.values()), 3),
            "models": {a: {"nDCG@10": round(100 * pq[a]["nDCG@10"].mean(), 1),
                           "MRR@10": round(100 * pq[a]["MRR@10"].mean(), 1)} for a in arms},
            "rank_order": arms, "pairs_nDCG@10": pairs, "pairs_MRR@10": pairs_mrr,
            "per_query_nDCG@10": {a: dict(zip([x["qid"] for x in items], np.round(pq[a]["nDCG@10"], 4).tolist()))
                                  for a in arms}}
    return out


def run_meta(runs):
    out = {}
    for arm, r in runs.items():
        lat = list(r["latency_s"].values())
        m = {"latency_median_ms": round(1000 * statistics.median(lat)),
             "latency_mean_ms": round(1000 * statistics.mean(lat)),
             "where": "API (OpenRouter)" if arm.startswith("jev") else "GPU RTX 5060 Ti 8GB"}
        for k in ("cost_usd_per_1000_queries", "calls_per_query"):
            if k in r["meta"]:
                m[k] = r["meta"][k]
        out[arm] = m
    return out


if __name__ == "__main__":
    items = data.load()
    runs = load_scores("miracl")
    qrels = {"original": {x["qid"]: [d["rel"] for d in x["docs"]] for x in items}}
    final = ROOT / "labels" / "final_qrels.json"
    if final.exists():
        fq = json.load(open(final, encoding="utf-8"))
        sub = [x for x in items if x["qid"] in fq["qrels"]]
        qrels = {"original": {x["qid"]: [d["rel"] for d in x["docs"]] for x in sub},
                 "relabelled": {x["qid"]: [int(fq["qrels"][x["qid"]].get(d["did"], d["rel"])) for d in x["docs"]]
                                for x in sub}}
        items = sub
    res = {"miracl": {"queries": len(items), "pairs": sum(len(x["docs"]) for x in items),
                      **evaluate_set("miracl", items, runs, qrels)},
           "miracl_runs": run_meta(runs)}
    json.dump(res, open(ROOT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for q in qrels:
        print(f"\n{q} qrels ({res['miracl'][q]['relevant_per_query']} relevant / query, {len(items)} queries)")
        for a, m in res["miracl"][q]["models"].items():
            print(f"  {NAMES[a]:28s} nDCG@10 {m['nDCG@10']:5.1f}  MRR@10 {m['MRR@10']:5.1f}  "
                  f"{res['miracl_runs'][a]['latency_median_ms']} ms/query")
        sig = sum(v["p"] < 0.05 for v in res["miracl"][q]["pairs_nDCG@10"].values())
        print(f"  significant pairs (nDCG@10, p<0.05): {sig}/{len(res['miracl'][q]['pairs_nDCG@10'])}")
