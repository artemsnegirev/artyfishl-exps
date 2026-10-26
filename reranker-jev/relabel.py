"""Relabel MIRACL-ru where the models actually look: the union of the models' top-10, judged blind by an LLM.

    .venv/Scripts/python reranker-jev/relabel.py pool [--queries N]   # labels/pool.jsonl + labels/batches/*.jsonl
    .venv/Scripts/python reranker-jev/relabel.py extend [--tag T_]    # pool += new arms' pairs, unlabelled -> labels/shards/T_shard_*
    .venv/Scripts/python reranker-jev/relabel.py final                # labels/final_qrels.json, dropped_positives.json

Which pooled pairs are judged: every passage no MIRACL annotator saw (`judged` is null), and every original
MIRACL positive (to catch false positives). Judged MIRACL negatives keep their 0 and are not re-asked.
The judge (Claude Opus, `prompt.md`, `labels/judge_brief.md`) sees only (query, passage) in shuffled order,
never the model that retrieved it or the old label; its answers are labels/A_*.jsonl.

Final label:
  unjudged by MIRACL   the judge's label
  MIRACL positive      stays 1 unless the judge says 0 (logged in dropped_positives.json)
  MIRACL negative      0

The pool was built with a few more arms than are compared now, so it also holds their top-10.
"""
import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LAB = ROOT / "labels"
sys.path.insert(0, str(ROOT))
import data  # noqa: E402
from evaluate import ARMS, load_scores, order  # noqa: E402

BATCH = 400


def read_jsonl(pattern):
    out = {}
    for f in sorted(glob.glob(str(LAB / pattern))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[(r["qid"], r["did"])] = r
    return out


def pool_pairs(items, runs):
    pairs = []
    for x in items:
        top = set()
        for arm in ARMS:
            top |= set(order(runs[arm]["scores"][x["qid"]])[:10])
        for k in sorted(top):
            d = x["docs"][k]
            if d["judged"] is None or d["judged"] == 1:
                pairs.append({"qid": x["qid"], "did": d["did"], "query": x["query"], "passage": d["text"]})
    return pairs


def cmd_pool(n):
    items, runs = data.load(), load_scores("miracl")
    missing = [a for a in ARMS if a not in runs]
    assert not missing, f"runs missing: {missing}"
    pairs = pool_pairs(items[:n], runs)
    random.Random(0).shuffle(pairs)
    (LAB / "batches").mkdir(parents=True, exist_ok=True)
    with open(LAB / "pool.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for i in range(0, len(pairs), BATCH):
        with open(LAB / "batches" / f"batch_{i // BATCH:02d}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs[i:i + BATCH]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    json.dump({"queries": n, "qids": [x["qid"] for x in items[:n]], "pairs": len(pairs)},
              open(LAB / "pool_meta.json", "w"), indent=1)
    print(f"pool: {n} queries, {len(pairs)} pairs, {-(-len(pairs) // BATCH)} batches of {BATCH}")


SHARD = 130


def cmd_extend(tag=""):
    """Add pairs of arms added after the first run and cut every pair that still lacks
    a judge label into shards of ~SHARD. Labelled pairs are never re-asked."""
    meta = json.load(open(LAB / "pool_meta.json"))
    items = {x["qid"]: x for x in data.load()}
    pool = read_jsonl("pool.jsonl")
    new = [p for p in pool_pairs([items[q] for q in meta["qids"]], load_scores("miracl"))
           if (p["qid"], p["did"]) not in pool]
    random.Random(1).shuffle(new)
    with open(LAB / "pool.jsonl", "a", encoding="utf-8") as f:
        for p in new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta["pairs"] = len(pool) + len(new)
    meta["added_by_extend"] = len(new)
    json.dump(meta, open(LAB / "pool_meta.json", "w"), indent=1)
    pool = read_jsonl("pool.jsonl")
    A = read_jsonl("A_*.jsonl")
    todo = [p for k, p in pool.items() if k not in A]
    out = LAB / "shards"
    out.mkdir(exist_ok=True)
    n = -(-len(todo) // SHARD)
    edges = [round(i * len(todo) / n) for i in range(n + 1)]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        with open(out / f"{tag}shard_{i:03d}.jsonl", "w", encoding="utf-8") as f:
            for p in todo[a:b]:
                f.write(json.dumps({"qid": p["qid"], "did": p["did"], "query": p["query"],
                                    "passage": p["passage"]}, ensure_ascii=False) + "\n")
    print(f"added {len(new)} pairs (pool {meta['pairs']}); {len(todo)} unlabelled -> {n} shards in {out}")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_final():
    meta = json.load(open(LAB / "pool_meta.json"))
    pool, A = read_jsonl("pool.jsonl"), read_jsonl("A_*.jsonl")
    lack = [k for k in pool if k not in A]
    assert not lack, f"{len(lack)} pooled pairs have no judge label"
    items = {x["qid"]: x for x in data.load()}
    qrels, dropped, stats, new_per_query = {}, [], collections.Counter(), {}
    for qid in meta["qids"]:
        x = items[qid]
        qrels[qid], new_per_query[qid] = {}, 0
        for d in x["docs"]:
            k = (qid, d["did"])
            label = 1 if d["judged"] == 1 or d["rel"] == 1 else 0
            if k in pool:
                la = int(A[k]["label"])
                if d["judged"] == 1:
                    if la == 0:
                        label = 0
                        dropped.append({"qid": qid, "did": d["did"], "query": x["query"], "passage": d["text"][:600],
                                        "reason": A[k].get("reason")})
                    stats[f"miracl_positive judge={la}"] += 1
                else:
                    label = la
                    stats[f"unjudged judge={la}"] += 1
                    new_per_query[qid] += la
            qrels[qid][d["did"]] = label
    counts = list(new_per_query.values())
    summary = {"queries": len(qrels), "pairs_judged": len(pool), "stats": dict(stats),
               "new_relevant_total": sum(counts),
               "new_relevant_per_query_mean": round(sum(counts) / len(counts), 3),
               "queries_with_new_relevant": sum(c > 0 for c in counts),
               "relevant_per_query_original": round(sum(sum(d["rel"] for d in items[q]["docs"]) for q in qrels)
                                                    / len(qrels), 3),
               "relevant_per_query_relabelled": round(sum(sum(v.values()) for v in qrels.values()) / len(qrels), 3)}
    json.dump({"summary": summary, "new_relevant_per_query": new_per_query, "qrels": qrels},
              open(LAB / "final_qrels.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(dropped, open(ROOT / "dropped_positives.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "extend", "final"])
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--tag", default="", help="extend: shard name prefix, so earlier shards are not overwritten")
    args = ap.parse_args()
    {"pool": lambda: cmd_pool(args.queries), "extend": lambda: cmd_extend(args.tag), "final": cmd_final}[args.cmd]()
