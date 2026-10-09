"""MRL / truncation experiment: FRIDA vs Qwen3-Embedding-0.6B on RuBQ Retrieval (ruMTEB).

Corpus and queries are encoded once at full dimension (cached to emb/*.npy),
then truncated to the first k components and L2-renormalized.
Metrics: nDCG@10 and recall@100 (binary qrels, MTEB-style).
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
EMB = ROOT / "emb"
EMB.mkdir(exist_ok=True)
DS = "ai-forever/rubq-retrieval"

MODELS = {
    "FRIDA": dict(
        name="ai-forever/FRIDA",
        q_prompt="search_query: ",
        d_prompt="search_document: ",
        dims=[1536, 1024, 768, 512, 384, 256, 128, 64, 32],
        doc_bs=32,  # 64 spills past 8 GB VRAM on the 5060 Ti
        kwargs={},
    ),
    "Qwen3-Embedding-0.6B": dict(
        name="Qwen/Qwen3-Embedding-0.6B",
        # instruction from the model card / config_sentence_transformers.json
        q_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        d_prompt="",
        dims=[1024, 768, 512, 384, 256, 128, 64, 32],
        doc_bs=16,
        kwargs=dict(tokenizer_kwargs={"padding_side": "left"}),
    ),
}
MAX_SEQ = 512


def load_rubq():
    p = lambda f: hf_hub_download(DS, f, repo_type="dataset")
    with zipfile.ZipFile(p("data/corpus.jsonl.zip")) as z, z.open("corpus.jsonl") as fh:
        corpus = [json.loads(l) for l in fh]
    queries = {(q := json.loads(l))["_id"]: q["text"] for l in open(p("data/queries.jsonl"), encoding="utf8")}
    qrels = {}
    for l in open(p("data/test.jsonl"), encoding="utf8"):
        r = json.loads(l)
        if r["score"] > 0:
            qrels.setdefault(r["query-id"], set()).add(r["corpus-id"])
    doc_ids = [d["_id"] for d in corpus]
    doc_texts = [(d["title"] + " " + d["text"]).strip() for d in corpus]
    q_ids = sorted(qrels, key=int)
    return doc_ids, doc_texts, q_ids, [queries[q] for q in q_ids], qrels


def encode(key, cfg, doc_texts, q_texts):
    dpath, qpath = EMB / f"{key}_corpus.npy", EMB / f"{key}_queries.npy"
    if dpath.exists() and qpath.exists():
        return np.load(dpath), np.load(qpath)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["name"], device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}, **cfg["kwargs"])
    model.max_seq_length = MAX_SEQ
    enc = lambda texts, prompt, bs: model.encode(
        texts, prompt=prompt, batch_size=bs, normalize_embeddings=False,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    q = enc(q_texts, cfg["q_prompt"], 64)
    d = enc(doc_texts, cfg["d_prompt"], cfg["doc_bs"])
    np.save(dpath, d)
    np.save(qpath, q)
    del model
    torch.cuda.empty_cache()
    return d, q


def evaluate(D, Q, doc_ids, q_ids, qrels, k_ndcg=10, k_rec=100):
    Dt = torch.from_numpy(D).cuda()
    Qt = torch.from_numpy(Q).cuda()
    Dt = torch.nn.functional.normalize(Dt, dim=-1)
    Qt = torch.nn.functional.normalize(Qt, dim=-1)
    top = torch.topk(Qt @ Dt.T, k_rec, dim=1).indices.cpu().numpy()
    disc = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    ndcgs, recs = [], []
    for qi, qid in enumerate(q_ids):
        rel = qrels[qid]
        hits = np.array([doc_ids[j] in rel for j in top[qi]], dtype=float)
        dcg = (hits[:k_ndcg] * disc).sum()
        idcg = disc[: min(len(rel), k_ndcg)].sum()
        ndcgs.append(dcg / idcg)
        recs.append(hits.sum() / len(rel))
    return float(np.mean(ndcgs)), float(np.mean(recs))


def main():
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_rubq()
    print(f"corpus={len(doc_ids)} queries={len(q_ids)} qrels={sum(map(len, qrels.values()))}")
    rows = []
    for key, cfg in MODELS.items():
        D, Q = encode(key, cfg, doc_texts, q_texts)
        full = D.shape[1]
        assert full == cfg["dims"][0], (key, full)
        for k in cfg["dims"]:
            ndcg, rec = evaluate(D[:, :k], Q[:, :k], doc_ids, q_ids, qrels)
            rows.append(dict(model=key, dim=k, ndcg10=ndcg, recall100=rec))
            print(f"{key:22s} dim={k:5d} nDCG@10={ndcg:.4f} R@100={rec:.4f}")
    df = pd.DataFrame(rows)
    for m in ("ndcg10", "recall100"):
        full = df.groupby("model")[m].transform("first")
        df[f"{m}_pct"] = 100 * df[m] / full
    df.to_csv(ROOT / "results.csv", index=False, float_format="%.4f")
    write_md(df)
    plot(df)


def write_md(df):
    dims = sorted(df.dim.unique(), reverse=True)
    lines = [
        "# MRL / truncation: RuBQ Retrieval (ruMTEB)",
        "",
        "Эмбеддинги в полной размерности обрезаются до первых k компонент и заново L2-нормируются.",
        "Метрики × 100. В скобках — % от полной размерности модели.",
        "",
        "| dim | FRIDA nDCG@10 | Qwen3-0.6B nDCG@10 | FRIDA R@100 | Qwen3-0.6B R@100 |",
        "|---:|---:|---:|---:|---:|",
    ]
    idx = df.set_index(["model", "dim"])

    def cell(model, d, m):
        if (model, d) not in idx.index:
            return "—"
        r = idx.loc[(model, d)]
        return f"{100 * r[m]:.1f} ({r[m + '_pct']:.0f}%)"

    for d in dims:
        lines.append(
            f"| {d} | {cell('FRIDA', d, 'ndcg10')} | {cell('Qwen3-Embedding-0.6B', d, 'ndcg10')} | "
            f"{cell('FRIDA', d, 'recall100')} | {cell('Qwen3-Embedding-0.6B', d, 'recall100')} |"
        )
    (ROOT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"FRIDA": "#2a6fdb", "Qwen3-Embedding-0.6B": "#e0663a"}
    labels = {"FRIDA": "FRIDA (без MRL)", "Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B (MRL)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, g in df.groupby("model"):
        g = g.sort_values("dim")
        kw = dict(marker="o", color=colors[model], label=labels[model], lw=2)
        axes[0].plot(g.dim, 100 * g.ndcg10, **kw)
        axes[1].plot(g.dim, 100 * g.recall100, **kw)
    ticks = [d for d in sorted(df.dim.unique()) if d not in (384, 768)]  # avoid label overlap
    for ax, title, ylabel in (
        (axes[0], "nDCG@10", "nDCG@10"),
        (axes[1], "Recall@100", "Recall@100"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("размерность (первые k компонент)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("RuBQ Retrieval (ruMTEB): обрезка эмбеддингов")
    fig.tight_layout()
    fig.savefig(ROOT / "mrl_rubq.png", dpi=160)


if __name__ == "__main__":
    main()
