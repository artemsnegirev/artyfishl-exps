"""Human check of the LLM relabelling on pairs where Jev pointwise and bge-reranker-v2-m3 disagree most.

    .venv/Scripts/python reranker-jev/human/human.py sample                  # human/sample.json, human/label.html
    .venv/Scripts/python reranker-jev/human/human.py analyze <labels.json>   # human/analysis.json

Sample (300 relabelled queries): "jev_up" = passage in Jev-pointwise top-5 but bge rank > 20; "bge_up" = bge
top-5 but Jev rank > 20 (ranks 1-based). At most one pair per query per side, 50 per side. Up to MIRACL_QUOTA
pairs per side are ones MIRACL annotators labelled (to compare humans with them), the rest MIRACL never saw;
within a query a seeded random pick, not the largest rank gap (that end is mostly junk and would leave few LLM
positives on the Jev side). Both
sides are shuffled together. label.html shows only the query and «Заголовок. текст» — no model, rank or label.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

HUM = Path(__file__).parent
ROOT = HUM.parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402
from evaluate import ARMS, NAMES, load_scores, metrics, order, signflip  # noqa: E402
from relabel import kappa, read_jsonl  # noqa: E402

PER_SIDE, TOP, FAR, SEED = 50, 5, 20, 0
MIRACL_QUOTA = {"jev_up": 15, "bge_up": 15}   # pairs MIRACL annotators labelled, to compare humans with them
norm = lambda s: " ".join(str(s).split())


def ranks(scores):
    return {k: r + 1 for r, k in enumerate(order(scores))}


def cmd_sample():
    fq = json.load(open(ROOT / "labels" / "final_qrels.json", encoding="utf-8"))["qrels"]
    items = [x for x in data.load() if x["qid"] in fq]
    runs = load_scores("miracl")
    A = read_jsonl("A_*.jsonl")
    rng = random.Random(SEED)
    cands = {"jev_up": [], "bge_up": []}
    for x in items:
        rj, rb = ranks(runs["jev-pw"]["scores"][x["qid"]]), ranks(runs["bge"]["scores"][x["qid"]])
        for side, mine, other in (("jev_up", rj, rb), ("bge_up", rb, rj)):
            ks = [k for k in mine if mine[k] <= TOP and other[k] > FAR]
            if ks:
                judged = [k for k in ks if x["docs"][k]["judged"] is not None]
                cands[side].append((x, rng.choice(judged) if judged else None,
                                    rng.choice([k for k in ks if k not in judged] or [None])))
    texts = data.titled_passages()
    sample = []
    for side, c in cands.items():
        rng.shuffle(c)
        with_j = [(k, x) for x, k, _ in c if k is not None][:MIRACL_QUOTA[side]]
        used = {x["qid"] for _, x in with_j}
        picked = with_j + [(k, x) for x, _, k in c if k is not None and x["qid"] not in used][:PER_SIDE - len(with_j)]
        print(f"{side}: {len(c)} queries with a candidate, {sum(t[1] is not None for t in c)} with a MIRACL-labelled one;"
              f" taking {len(with_j)} MIRACL-labelled + {len(picked) - len(with_j)} unlabelled")
        for k, x in picked:
            d = x["docs"][k]
            key = (x["qid"], d["did"])
            lab = lambda J: int(J[key]["label"]) if key in J else None
            sample.append({"qid": x["qid"], "did": d["did"], "query": norm(x["query"]), "passage": texts[d["did"]],
                           "side": side,
                           "jev_rank": ranks(runs["jev-pw"]["scores"][x["qid"]])[k],
                           "bge_rank": ranks(runs["bge"]["scores"][x["qid"]])[k],
                           "miracl_label": d["judged"], "judge_label": lab(A),
                           "final_label": int(fq[x["qid"]][d["did"]])})
    rng.shuffle(sample)
    for i, s in enumerate(sample, 1):
        s["id"] = i
    sample = [{"id": s.pop("id"), **s} for s in sample]
    json.dump(sample, open(HUM / "sample.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    page = (HUM / "label_template.html").read_text(encoding="utf-8").replace(
        "/*PAIRS*/[]", json.dumps([{"id": s["id"], "q": s["query"], "p": s["passage"]} for s in sample],
                                  ensure_ascii=False).replace("</", "<\\/"))
    (HUM / "label.html").write_text(page, encoding="utf-8")
    for side in cands:
        s = [r for r in sample if r["side"] == side]
        print(f"{side}: {len(s)} pairs, MIRACL-labelled {sum(r['miracl_label'] is not None for r in s)}, "
              f"final LLM label 1: {sum(r['final_label'] for r in s)}")


def ndcg_table(items, runs, qrels):
    return {a: np.array([metrics([qrels[x["qid"]][k] for k in order(runs[a]["scores"][x["qid"]])],
                                 sum(qrels[x["qid"]]))["nDCG@10"] for x in items]) for a in ARMS}


def cmd_analyze(path):
    sample = json.load(open(HUM / "sample.json", encoding="utf-8"))
    raw = json.load(open(path, encoding="utf-8"))
    raw = raw.get("labels", raw) if isinstance(raw, dict) else raw
    human = {int(r["id"]): r["label"] for r in raw}
    # corrections Arty gave in chat after the interim check; applied only where the file still has the old answer
    for c in json.load(open(HUM / "corrections.json", encoding="utf-8")):
        if human.get(c["id"]) == c["old"]:
            human[c["id"]] = c["new"]
    code = {"relevant": 1, "not_relevant": 0}
    out = {"labelled": len(human), "unsure": sum(v == "unsure" for v in human.values()), "sides": {}}
    rows = [{**s, "human": code.get(human.get(s["id"]))} for s in sample]
    for side in ("all", "jev_up", "bge_up"):
        r = [x for x in rows if side == "all" or x["side"] == side]
        h = [x for x in r if x["human"] is not None]
        m = [x for x in h if x["miracl_label"] is not None]
        o = {"pairs": len(r), "human_labelled": len(h), "unsure": sum(human.get(x["id"]) == "unsure" for x in r),
             "human_relevant_share": round(sum(x["human"] for x in h) / len(h), 3) if h else None,
             "llm_relevant_share": round(sum(x["final_label"] for x in h) / len(h), 3) if h else None,
             "agree_human_llm": round(sum(x["human"] == x["final_label"] for x in h) / len(h), 3) if h else None,
             "kappa_human_llm": round(kappa([x["human"] for x in h], [x["final_label"] for x in h]), 3) if h else None,
             "confusion_human_x_llm": {f"h{a}_llm{b}": sum(x["human"] == a and x["final_label"] == b for x in h)
                                       for a in (0, 1) for b in (0, 1)},
             "miracl_labelled": len(m)}
        # pairs the judge actually saw (MIRACL negatives were not re-asked: their final label is MIRACL's 0)
        j = [x for x in h if x["judge_label"] is not None]
        if j:
            o["judge_seen"] = len(j)
            o["agree_human_judge"] = sum(x["human"] == x["judge_label"] for x in j)
            if len({x["human"] for x in j} | {x["judge_label"] for x in j}) > 1:
                o["kappa_human_judge"] = round(kappa([x["human"] for x in j], [x["judge_label"] for x in j]), 3)
        if m:
            o["agree_human_miracl"] = round(sum(x["human"] == x["miracl_label"] for x in m) / len(m), 3)
            o["kappa_human_miracl"] = round(kappa([x["human"] for x in m], [x["miracl_label"] for x in m]), 3)
            o["confusion_human_x_miracl"] = {f"h{a}_miracl{b}": sum(x["human"] == a and x["miracl_label"] == b
                                                                    for x in m) for a in (0, 1) for b in (0, 1)}
        out["sides"][side] = o

    # nDCG@10 on the 300 queries: relabelled qrels, then the same with human labels overriding the sampled pairs
    fq = json.load(open(ROOT / "labels" / "final_qrels.json", encoding="utf-8"))["qrels"]
    items = [x for x in data.load() if x["qid"] in fq]
    runs = load_scores("miracl")
    relab = {x["qid"]: [int(fq[x["qid"]].get(d["did"], d["rel"])) for d in x["docs"]] for x in items}
    hum = {q: list(v) for q, v in relab.items()}
    idx = {x["qid"]: {d["did"]: k for k, d in enumerate(x["docs"])} for x in items}
    for x in rows:
        if x["human"] is not None:
            hum[x["qid"]][idx[x["qid"]][x["did"]]] = x["human"]
    orig = {x["qid"]: [d["rel"] for d in x["docs"]] for x in items}
    sub = [x for x in items if x["qid"] in {r["qid"] for r in rows}]
    rng = np.random.default_rng(0)
    out["ndcg10"] = {}
    for name, its in (("all_300", items), ("sampled_queries", sub)):
        t = {q: ndcg_table(its, runs, qr) for q, qr in (("original", orig), ("relabelled", relab), ("human", hum))}
        res = {"queries": len(its),
               "models": {a: {q: round(100 * t[q][a].mean(), 2) for q in t} for a in ARMS}}
        for q in t:
            d = t[q]["jev-pw"] - t[q]["bge"]
            res[f"jev-pw_minus_bge_{q}"] = {"diff": round(100 * d.mean(), 2), "p": signflip(d, rng)}
        g_rel = res["jev-pw_minus_bge_relabelled"]["diff"] - res["jev-pw_minus_bge_original"]["diff"]
        g_hum = res["jev-pw_minus_bge_human"]["diff"] - res["jev-pw_minus_bge_original"]["diff"]
        res["gain_vs_bge_from_relabelling"] = round(g_rel, 2)
        res["gain_vs_bge_with_human_overrides"] = round(g_hum, 2)
        res["gain_surviving_share"] = round(g_hum / g_rel, 3) if g_rel else None
        out["ndcg10"][name] = res
    out["disputed"] = [{k: x[k] for k in ("id", "qid", "side", "jev_rank", "bge_rank", "miracl_label", "judge_label",
                                          "final_label", "human")} | {"query": x["query"],
                                                                                              "passage": x["passage"][:300]}
                       for x in rows if x["human"] is not None and x["human"] != x["final_label"]]
    json.dump(out, open(HUM / "analysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disputed"}, ensure_ascii=False, indent=1))
    print(f"{len(out['disputed'])} pairs where human != final LLM label (in analysis.json)")


if __name__ == "__main__":
    {"sample": cmd_sample, "analyze": lambda: cmd_analyze(sys.argv[2])}[sys.argv[1]]()
