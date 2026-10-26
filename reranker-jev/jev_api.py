"""TypeSafe Jev over OpenRouter's decisions endpoint, with an append-only disk cache keyed by request hash.

Wire format and retry policy follow frida-static/scripts/bench_jev.py (arm_typesafe_jev). The key is read
from frida-static/.env and never printed. Every response is stored with its wall-clock latency, so a rerun
costs nothing and the latency/cost numbers come from the same file.
"""
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
PRICE_IN = 0.042e-6   # $ per input token; fitted exactly on 16.8k cached bills, output tokens are free
ROOT = Path(__file__).parent
CACHE = ROOT / "cache" / "jev_responses.jsonl"
ENV = ROOT.parents[1] / "frida-static" / ".env"


def _key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and ENV.exists():
        for line in ENV.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not found")
    return key


def body_hash(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _call(body, key):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(7):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                              "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, json.load(r), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            if e.code != 429 and e.code < 500:
                try:
                    return e.code, json.loads(text), 0.0
                except ValueError:
                    return e.code, {"error": text[:500]}, 0.0
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(2 ** attempt)
    return 0, {"error": "gave up after 7 attempts"}, 0.0


class Jev:
    def __init__(self):
        self.cache = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["hash"]] = rec
        self.lock = threading.Lock()

    def ask_many(self, bodies, workers=32, label="jev"):
        """Returns cache records ({"response", "latency"}) in order; failed ones are None."""
        bodies = [{"model": MODEL, **b} for b in bodies]
        hashes = [body_hash(b) for b in bodies]
        todo = sorted({h: i for i, h in enumerate(hashes) if h not in self.cache}.values())
        if todo:
            key, done, failed = _key(), [0], {}
            t0 = time.time()
            print(f"  {label}: {len(bodies) - len(todo)} cached, asking {len(todo)}", flush=True)

            def ask(i):
                status, parsed, lat = _call(bodies[i], key)
                with self.lock:
                    done[0] += 1
                    if status == 200 and isinstance(parsed.get("answers"), dict):
                        rec = {"hash": hashes[i], "response": parsed, "latency": lat}
                        self.cache[hashes[i]] = rec
                        with CACHE.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        failed[i] = (status, json.dumps(parsed, ensure_ascii=False)[:300])
                    if done[0] % 500 == 0 or done[0] == len(todo):
                        print(f"  {label}: {done[0]}/{len(todo)}, {len(failed)} failed, "
                              f"{done[0] / (time.time() - t0):.1f} req/s", flush=True)

            CACHE.parent.mkdir(exist_ok=True)
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(ask, todo))
            if failed:
                first = next(iter(failed.values()))
                print(f"  {label}: {len(failed)} FAILED, first: HTTP {first[0]} {first[1]}", flush=True)
                hard = [f for f in failed.values() if f[0] != 400]
                if hard:
                    raise SystemExit(f"{label}: {len(hard)} requests failed for a reason that is not the "
                                     "item's; answers so far are cached, rerun to resume")
        return [self.cache.get(h) for h in hashes]
