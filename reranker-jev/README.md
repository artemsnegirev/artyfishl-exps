# reranker-jev

Насколько Jev хорош как реранкер? MIRACL-ru dev из MTEB: 300 запросов × 100 кандидатов, одни и те же
кандидаты для всех моделей. По исходной разметке MIRACL Jev последний, но в его топ-10 люди оценили только
31 % пассажей. Топ-10 всех моделей доразмечен LLM-судьёй, судья сверен с ручной разметкой.

| Модель | nDCG@10, исходная разметка | nDCG@10, доразмеченная |
|---|---|---|
| Jev pointwise (API) | 80.5 | **91.3** |
| Jev listwise (API) | 82.1 | 87.6 |
| FRIDA-Decisions | 85.7 | 81.6 |
| BGE-reranker-v2-m3 | **87.4** | 79.0 |
| KaLM-Reranker-V1-Nano-R2 | 86.6 | 78.8 |
| Qwen3-Reranker-0.6B | 82.0 | 75.2 |

MRR@10, латентность, стоимость Jev и попарная значимость (sign-flip, 10k перестановок) — в `results.json`.

**Доразметка.** Пул — объединение топ-10 моделей, 6 752 пары. Пассажи, которые MIRACL не видел, и все
позитивы MIRACL вслепую размечает Claude Opus по [`prompt.md`](prompt.md) («есть ли в пассаже ответ»);
негативы MIRACL остаются 0. Релевантных на запрос стало 4.86 вместо 2.36.

**Проверка судьи.** 50 спорных пар (топ-5 у Jev и ниже 20-го места у BGE, и наоборот), размечены вручную
вслепую: с судьёй совпадение 35 из 37, с аннотаторами MIRACL — 11 из 11. Если снять случайную половину
добавленных судьёй позитивов, Jev pointwise всё равно первый: 84.4 против 79.6 у FRIDA-Decisions.

## Запуск

```
.venv/Scripts/python reranker-jev/data.py                  # набор -> data/miracl.json
.venv/Scripts/python reranker-jev/run.py --arm all         # скоры -> scores/miracl/<модель>.json
.venv/Scripts/python reranker-jev/relabel.py final         # метки судьи -> labels/final_qrels.json
.venv/Scripts/python reranker-jev/evaluate.py              # -> results.json
.venv/Scripts/python reranker-jev/drop_half.py             # -> drop_half.json
.venv/Scripts/python reranker-jev/figures.py               # -> figures/
```

Jev идёт через OpenRouter (`OPENROUTER_API_KEY`), остальные модели — на GPU.

## Файлы

- `labels/` — пул, ответы судьи с причинами (`A_*.jsonl`), задание судье, итоговые qrels
- `human/` — страница для ручной разметки, выборка, ручные метки и их сверка с судьёй
- `scores/`, `results.json`, `drop_half.json`, `dropped_positives.json` — результаты
