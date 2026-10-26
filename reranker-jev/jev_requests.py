"""The requests the Jev arms send."""

STATE = "Запрос: {query}\nФрагмент: {passage}"
RELEVANCE = {"type": "noul", "instructions": "Отвечает ли фрагмент на запрос?",
             "criteria": {"true": "Фрагмент отвечает на запрос.", "false": "Фрагмент не отвечает на запрос."}}
LIST_STATE = "Запрос: {query}"
LIST_Q = "Какой фрагмент лучше всего отвечает на запрос?"
BATCH = 25   # listwise candidates per call: MIRACL's 100 go in 4


def pointwise(query, passage):
    return {"state": STATE.format(query=query, passage=passage), "questions": {"relevance": RELEVANCE}}


def listwise(query, passages):
    return {"state": LIST_STATE.format(query=query),
            "questions": {"best": {"type": "choice", "instructions": LIST_Q,
                                   "criteria": {str(i): p for i, p in enumerate(passages)}}}}


def batches(n, size=BATCH):
    """Split n candidates into near-equal contiguous batches of at most `size`."""
    k = -(-n // size)
    edges = [round(i * n / k) for i in range(k + 1)]
    return [range(a, b) for a, b in zip(edges, edges[1:])]
