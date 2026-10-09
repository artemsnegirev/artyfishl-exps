# artyfishl-exps

Эксперименты для Telegram-канала [@artyfishl](https://t.me/artyfishl).
В каждой подпапке лежат скрипт, результаты (csv/md) и графики. Кэши эмбеддингов в git не попадают.

## Окружение

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements.txt
```

## Эксперименты

| Папка | Что | Запуск |
|---|---|---|
| [mrl/](mrl/) | Обрезка эмбеддингов (MRL): FRIDA vs Qwen3-Embedding-0.6B на RuBQ Retrieval (ruMTEB). nDCG@10 и recall@100 от размерности 32…1536 | `.venv/Scripts/python mrl/run_mrl.py` |
