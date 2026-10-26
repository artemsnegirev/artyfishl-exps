"""Every reranker on the same rows: writes scores/miracl/<arm>.json with one score per candidate, in data order.

    .venv/Scripts/python reranker-jev/run.py --arm all

Arms (higher score = more relevant):
  jev-pw        TypeSafe Jev, one `noul` per (query, passage): P(yes), rounded to 0.01 by the API
  jev-list      TypeSafe Jev, one `choice` per query over the candidates (it has no ranking type). Pools over
                25 go in near-equal batches; then the top 25 by batch probability are asked again together,
                and they rank above the rest by the second answer. P rounded to 0.01, so many zeros tie.
  fd            FRIDA-Decisions, one `ranking` per query over all 100 candidates «Заголовок. текст»: raw margins
  bge           BAAI/bge-reranker-v2-m3 cross-encoder, fp16, max 512 tokens
  qwen3-0.6b    Qwen/Qwen3-Reranker-0.6B, the model card's prompt, P(yes) at the last token, bf16, max 1024
  kalm-nano-r2  KaLM-Embedding/KaLM-Reranker-V1-Nano-R2, sentence-transformers CrossEncoder as on the card:
                bf16, chunk_size=4, the card's default "retrieval" instruction, P(yes)

Latency is per query. GPU arms score one query's candidates as one batch, synchronised, so it is the wall
time of that query on an RTX 5060 Ti 8 GB. API arms fire a query's calls concurrently: latency is the slowest
call of the query (plus stage 2 for jev-list).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "frida-decisions-main"))
import data  # noqa: E402
import jev_requests as R  # noqa: E402

ARMS = ["jev-pw", "jev-list", "fd", "bge", "qwen3-0.6b", "kalm-nano-r2"]


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def per_query(items, fn):
    """fn(item) -> list of scores; returns scores and per-query latency (seconds)."""
    scores, lat = {}, {}
    for k, x in enumerate(items):
        _sync()
        t0 = time.perf_counter()
        scores[x["qid"]] = [float(s) for s in fn(x)]
        _sync()
        lat[x["qid"]] = time.perf_counter() - t0
        if k % 100 == 0:
            print(f"    {k}/{len(items)}", flush=True)
    return scores, lat


# ------------------------------------------------------------------ API arms
def _usage(recs):
    return sum(r["response"]["usage"]["input_tokens"] for r in recs if r), \
        sum(float(r["response"]["usage"].get("cost") or 0) for r in recs if r)


def arm_jev_pw(items):
    from jev_api import Jev
    jev = Jev()
    bodies = [R.pointwise(x["query"], d["text"]) for x in items for d in x["docs"]]
    recs = jev.ask_many(bodies, label="jev-pw")
    scores, lat, i = {}, {}, 0
    for x in items:
        rs = recs[i:i + len(x["docs"])]
        i += len(x["docs"])
        scores[x["qid"]] = [r["response"]["answers"]["relevance"]["noul"] if r else 0.0 for r in rs]
        lat[x["qid"]] = max(r["latency"] for r in rs if r)
    tokens, cost = _usage(recs)
    return scores, lat, {"input_tokens": tokens, "cost_usd": cost, "calls": len(bodies),
                         "unanswered_scored_0": sum(r is None for r in recs)}


def arm_jev_list(items):
    from jev_api import Jev
    jev = Jev()

    def probs(rec, n):
        p = rec["response"]["answers"]["best"]["probabilities"] if rec else {}
        return [float(p.get(str(k), 0.0)) for k in range(n)]

    plan1 = [(j, b) for j, x in enumerate(items) for b in R.batches(len(x["docs"]))]   # (item index, batch)
    recs1 = jev.ask_many([R.listwise(items[j]["query"], [items[j]["docs"][k]["text"] for k in b])
                          for j, b in plan1], label="jev-list stage 1")
    p1 = {j: [0.0] * len(x["docs"]) for j, x in enumerate(items)}
    lat1 = {j: 0.0 for j in p1}
    for (j, b), rec in zip(plan1, recs1):
        for k, p in zip(b, probs(rec, len(b))):
            p1[j][k] = p
        lat1[j] = max(lat1[j], rec["latency"] if rec else 0.0)
    plan2 = []
    for j, x in enumerate(items):
        if len(R.batches(len(x["docs"]))) > 1:
            top = sorted(range(len(x["docs"])), key=lambda k: -p1[j][k])[:R.BATCH]  # stable: ties keep data order
            plan2.append((j, top))
    recs2 = jev.ask_many([R.listwise(items[j]["query"], [items[j]["docs"][k]["text"] for k in top])
                          for j, top in plan2], label="jev-list stage 2")
    scores = {x["qid"]: list(p1[j]) for j, x in enumerate(items)}
    lat = {x["qid"]: lat1[j] for j, x in enumerate(items)}
    for (j, top), rec in zip(plan2, recs2):
        qid = items[j]["qid"]
        for k, p in zip(top, probs(rec, len(top))):
            scores[qid][k] = 1.0 + p + 1e-3 * p1[j][k]
        lat[qid] += rec["latency"] if rec else 0.0
    recs = recs1 + recs2
    tokens, cost = _usage(recs)
    return scores, lat, {"input_tokens": tokens, "cost_usd": cost, "calls": len(recs),
                         "unanswered": sum(r is None for r in recs)}


# ------------------------------------------------------------------ GPU arms
FD_QUESTION = "Какой из фрагментов отвечает на запрос?"


def arm_fd(items):
    from frida_decisions import Judge
    judge = Judge.from_pretrained("ai-forever/FRIDA-Decisions", device="cuda", state_max=512)
    texts = data.titled_passages()

    def fn(x):
        req = {"state": f"Запрос: {data.norm(x['query'])}",
               "questions": {"relevance": {"type": "ranking", "instructions": FD_QUESTION,
                                                "criteria": {str(k + 1): texts[d["did"]]
                                                             for k, d in enumerate(x["docs"])}}}}
        m = judge(req)["margins"]["relevance"]
        return [m[str(k + 1)] for k in range(len(x["docs"]))]
    fn(items[0])
    s, l = per_query(items, fn)
    return s, l, {"dtype": str(judge.dtype), "question": FD_QUESTION}


def arm_bge(items):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda",
                         model_kwargs={"torch_dtype": "float16"})

    def fn(x):
        return model.predict([(x["query"], d["text"]) for d in x["docs"]], batch_size=50, activation_fn=None)
    fn(items[0])
    s, l = per_query(items, fn)
    return s, l, {"dtype": "fp16", "max_length": 512}


QWEN_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
               "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def arm_qwen3(items, name="Qwen/Qwen3-Reranker-0.6B", max_len=1024):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok.encode(QWEN_PREFIX, add_special_tokens=False), tok.encode(QWEN_SUFFIX, add_special_tokens=False)

    @torch.inference_mode()
    def fn(x, bs=25):
        texts = [f"<Instruct>: {QWEN_TASK}\n<Query>: {x['query']}\n<Document>: {d['text']}" for d in x["docs"]]
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=False, truncation="longest_first",
                      max_length=max_len - len(pre) - len(suf), return_attention_mask=False)
            ids = [pre + t + suf for t in enc["input_ids"]]
            batch = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, no], logits[:, yes]], 1).float().log_softmax(1)
            out += pair[:, 1].exp().cpu().tolist()
        return out
    fn(items[0])
    s, l = per_query(items, fn)
    return s, l, {"dtype": "bf16", "max_length": max_len, "instruction": QWEN_TASK}


KALM_REV = "32b242bb7fd7ad6fd404154c09560d249ffe017f"   # remote code read before pinning


def arm_kalm(items, name="KaLM-Embedding/KaLM-Reranker-V1-Nano-R2"):
    import torch
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(name, revision=KALM_REV, trust_remote_code=True, device="cuda",
                         model_kwargs={"dtype": torch.bfloat16, "chunk_size": 4})

    def fn(x):
        return model.predict([(x["query"], d["text"]) for d in x["docs"]], batch_size=50)
    fn(items[0])
    s, l = per_query(items, fn)
    return s, l, {"dtype": "bf16", "chunk_size": 4, "revision": KALM_REV, "prompt": model.default_prompt_name}


RUN = {"jev-pw": arm_jev_pw, "jev-list": arm_jev_list, "fd": arm_fd, "bge": arm_bge, "qwen3-0.6b": arm_qwen3,
       "kalm-nano-r2": arm_kalm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", help="comma-separated, or all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    items = data.load()
    out_dir = ROOT / "scores" / "miracl"
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS if args.arm == "all" else args.arm.split(","):
        path = out_dir / f"{arm}.json"
        if path.exists() and not args.force:
            print(f"{arm}: {path.name} exists, skipped")
            continue
        print(f"{arm} ({len(items)} queries)", flush=True)
        t0 = time.time()
        scores, lat, meta = RUN[arm](items)
        for q, v in scores.items():
            assert len(v) == len(next(x for x in items if x["qid"] == q)["docs"]) and all(map(math.isfinite, v)), q
        meta["wall_s"] = time.time() - t0
        json.dump({"arm": arm, "meta": meta, "latency_s": lat, "scores": scores}, open(path, "w", encoding="utf-8"))
        print(f"{arm}: done in {meta['wall_s']:.0f}s"
              + (f", ${meta['cost_usd']:.3f}" if "cost_usd" in meta else ""), flush=True)
