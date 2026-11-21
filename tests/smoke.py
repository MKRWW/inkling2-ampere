import os
import json, time, urllib.request, base64, struct, zlib

URL = os.environ.get("INKLING_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
M = "inkling-2-mini"


def post(body, timeout=300):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t


def show(name, d, dt):
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    print(f"== {name}: {dt:.1f}s, usage {u}, finish {d['choices'][0]['finish_reason']}")
    if m.get("reasoning") or m.get("reasoning_content"):
        print("   reasoning:", (m.get("reasoning") or m.get("reasoning_content"))[:200].replace("\n", " "))
    print("   content:", (m.get("content") or "")[:300].replace("\n", " "))
    if m.get("tool_calls"):
        print("   tool_calls:", json.dumps(m["tool_calls"])[:300])


# 1. Rechenaufgabe mit bekanntem Ergebnis
d, dt = post({"model": M, "messages": [{"role": "user", "content": "Was ist 17 * 23? Antworte nur mit der Zahl."}],
              "max_tokens": 2000, "temperature": 1.0, "top_p": 1.0})
show("17*23", d, dt)

# 2. Tool-Call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Aktuelles Wetter einer Stadt",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, dt = post({"model": M, "messages": [{"role": "user", "content": "Wie ist das Wetter in Koeln?"}],
              "tools": tools, "max_tokens": 2000})
show("tool", d, dt)

# 3. Durchsatz: lange Ausgabe
d, dt = post({"model": M, "messages": [{"role": "user", "content": "Schreibe eine Python-Funktion, die einen LRU-Cache mit TTL implementiert, mit Docstrings und Tests."}],
              "max_tokens": 1500, "temperature": 1.0})
u = d["usage"]
print(f"== durchsatz: {u['completion_tokens']} Token in {dt:.1f}s = {u['completion_tokens'] / dt:.1f} t/s (inkl. Prefill)")

# 4. Bild: 64x64 PNG, links rot, rechts blau
def png(w, h):
    raw = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w)) for _ in range(h))
    def chunk(t, data):
        return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
img = base64.b64encode(png(64, 64)).decode()
try:
    d, dt = post({"model": M, "max_tokens": 1500, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Welche zwei Farben siehst du, und wo?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}]})
    show("bild", d, dt)
except Exception as e:
    print("== bild FEHLER", e)
