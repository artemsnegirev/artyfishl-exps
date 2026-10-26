"""The fixed evaluation set: every arm reads data/miracl.json, so all models see the same rows in the same order.

MTEB MIRACLReranking (ru), dev: 100 candidates per query, binary qrels. 300 queries, seed 0, sampled among
queries with at least one relevant candidate (a query without one scores 0 for every model and only dilutes
the differences). Each candidate also carries `judged`: its label in the original MIRACL dev qrels (0/1), or
null when no MIRACL annotator ever saw it -- matched by passage text, since the MTEB ids are renumbered.

Candidate order is shuffled with the seed: the shipped pools are in retriever order, and a tie-heavy
scorer would otherwise inherit it.
"""
import collections
import gzip
import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 0
N_QUERIES = 300
MTEB = dict(repo="mteb/MIRACLReranking", rev="ab6f54eff185a84bc1f6ab96b56bc7df87433228", split="dev",
            files={"queries": "ru-queries", "corpus": "ru-corpus", "pools": "ru-top_ranked", "qrels": "ru-qrels"})
MIRACL = dict(repo="miracl/miracl", rev="5be20db9509754dadad47689368639fcec739c00", sub="miracl-v1.0-ru")
CORPUS = dict(repo="miracl/miracl-corpus", rev="d921ec7e349ce0d28daf30b2da9da5ee698bef0d", sub="miracl-corpus-v1.0-ru")
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def original_judgments():
    """{query text: {passage text: label}} from the original MIRACL ru dev qrels."""
    path = Path(snapshot_download(MIRACL["repo"], repo_type="dataset", revision=MIRACL["rev"],
                                  allow_patterns=[f"{MIRACL['sub']}/qrels/*dev*", f"{MIRACL['sub']}/topics/*dev*"]))
    base = path / MIRACL["sub"]
    topics = dict(l.rstrip("\n").split("\t") for l in open(base / "topics/topics.miracl-v1.0-ru-dev.tsv", encoding="utf-8"))
    qrels = collections.defaultdict(dict)
    for line in open(base / "qrels/qrels.miracl-v1.0-ru-dev.tsv"):
        q, _, d, s = line.split()
        qrels[q][d] = int(s)
    need = {d for v in qrels.values() for d in v}
    corpus = Path(snapshot_download(CORPUS["repo"], repo_type="dataset", revision=CORPUS["rev"],
                                    allow_patterns=[f"{CORPUS['sub']}/*"])) / CORPUS["sub"]
    texts = {}
    for f in sorted(corpus.glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            i = line.find('"docid": "') + 10
            if line[i:line.find('"', i)] in need:
                r = json.loads(line)
                texts[r["docid"]] = r["text"]
    return {norm(topics[q]): {norm(texts[d]): s for d, s in v.items()} for q, v in qrels.items()}


def build_miracl():
    s = MTEB
    path = Path(snapshot_download(s["repo"], repo_type="dataset", revision=s["rev"],
                                  allow_patterns=[f"{v}/*" for v in s["files"].values()]))
    part = {k: pd.read_parquet(next((path / v).glob(f"{s['split']}-*.parquet"))) for k, v in s["files"].items()}
    queries = dict(zip(part["queries"]["_id"], part["queries"]["text"]))
    c = part["corpus"]
    corpus = {i: (f"{t}\n{x}" if t else x) for i, t, x in zip(c["_id"], c["title"].fillna(""), c["text"])}
    rel = {}
    r = part["qrels"]
    for q, d, score in zip(r["query-id"], r["corpus-id"], r["score"]):
        if score > 0:
            rel.setdefault(q, set()).add(d)
    rng = random.Random(SEED)
    pools = [(q, list(ids)) for q, ids in zip(part["pools"]["query-id"], part["pools"]["corpus-ids"])
             if rel.get(q, set()) & set(ids)]
    dropped = len(part["pools"]) - len(pools)
    rng.shuffle(pools)
    pools = pools[:N_QUERIES]
    judged = original_judgments()
    out = []
    for q, ids in pools:
        rng.shuffle(ids)
        jq = judged[norm(queries[q])]
        docs = []
        for d in ids:
            t = norm(corpus[d])
            # the MTEB text is "<title> <passage>"; MIRACL's is the passage alone
            hit = next((v for jt, v in jq.items() if t.endswith(jt)), None)
            docs.append({"did": d, "text": corpus[d], "rel": int(d in rel[q]), "judged": hit})
        out.append({"qid": q, "query": queries[q], "docs": docs})
    flat = [d for x in out for d in x["docs"]]
    meta = {"source": f"{s['repo']}@{s['rev'][:8]} {s['split']}", "queries": len(out), "pairs": len(flat),
            "relevant_pairs": sum(d["rel"] for d in flat), "dropped_no_relevant": dropped, "seed": SEED,
            "miracl_judged": collections.Counter(f"rel={d['rel']} judged={d['judged']}" for d in flat)}
    return out, meta


def titled_passages():
    """{did: «<Заголовок>. <текст>»} (just the text when it already starts with the title), the form shown to
    FRIDA-Decisions and on the human labelling page."""
    s = MTEB
    path = Path(snapshot_download(s["repo"], repo_type="dataset", revision=s["rev"],
                                  allow_patterns=[f"{s['files']['corpus']}/*"]))
    c = pd.read_parquet(next((path / s["files"]["corpus"]).glob(f"{s['split']}-*.parquet")))
    out = {}
    for i, t, x in zip(c["_id"], c["title"].fillna(""), c["text"]):
        t, x = norm(str(t)), norm(str(x))
        out[i] = x if (not t or x.startswith(t)) else f"{t}. {x}"
    return out


def load():
    return json.load(open(DATA / "miracl.json", encoding="utf-8"))["items"][:N_QUERIES]


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    items, meta = build_miracl()
    json.dump({"meta": meta, "items": items}, open(DATA / "miracl.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(meta)
