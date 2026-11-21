import os
import json, time, urllib.request, sys

URL = os.environ.get("INKLING_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
n_words = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
words = ("Der Router verteilt Anfragen zwischen zwei Modellen und merkt sich Gespraeche. " * (n_words // 10 + 1)).split()[:n_words]
text = " ".join(f"{w}{i}" if i % 97 == 0 else w for i, w in enumerate(words))
body = {"model": "inkling-2-mini", "stream": True, "max_tokens": 200, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": text + "\n\nWie viele Woerter mit angehaengter Zahl stehen ungefaehr im Text? Kurz."}]}
req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); first = None; usage = None; n = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            if (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")) and first is None:
                first = time.time()
            n += 1
t1 = time.time()
pt = usage["prompt_tokens"] if usage else None
ct = usage["completion_tokens"] if usage else None
print(f"prompt {pt} Token, TTFT {first - t0:.1f}s = {pt / (first - t0):.0f} t/s Prefill, "
      f"Decode {ct} Token in {t1 - first:.1f}s = {ct / (t1 - first):.1f} t/s")
