"""Nadel im Heuhaufen: drei Fakten an verschiedenen Stellen eines langen Textes, danach Abfrage aller drei."""
import os
import json, random, sys, time, urllib.request

URL = os.environ.get("INKLING_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"
target_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
random.seed(7)
filler = [
    "Im Lager wurden die Kisten nach Farbe sortiert und am Abend gezaehlt.",
    "Die Maschine lief ruhig, nur das Lueftergeraeusch war zu hoeren.",
    "Am Bahnhof warteten Reisende auf den verspaeteten Regionalzug.",
    "Der Bericht fasste die Messwerte der letzten Woche zusammen.",
    "Im Garten wuchsen Tomaten, Bohnen und ein paar Sonnenblumen.",
]
needles = {
    0.12: ("Der Code fuer das blaue Tor lautet 4711-KESTREL.", "blaue Tor", "4711-KESTREL"),
    0.55: ("Die Kantine schliesst donnerstags um 13:37 Uhr.", "Kantine donnerstags", "13:37"),
    0.91: ("Der Ersatzschluessel liegt unter dem gruenen Kaktus.", "Ersatzschluessel", "Kaktus"),
}
# grob 1 Token je ~4 Zeichen deutscher Text mit diesem Tokenizer; wird ueber usage korrigiert
n_sent = target_tokens // 14
sents = [random.choice(filler) for _ in range(n_sent)]
for frac, (fact, _, _) in needles.items():
    sents.insert(int(frac * len(sents)), fact)
text = " ".join(sents)
q = ("Beantworte anhand des Textes drei Fragen, je eine Zeile: 1) Wie lautet der Code fuer das blaue Tor? "
     "2) Wann schliesst die Kantine donnerstags? 3) Wo liegt der Ersatzschluessel?")
body = {"model": "inkling-2-mini", "max_tokens": 3000, "temperature": 1.0, "top_p": 1.0,
        "messages": [{"role": "user", "content": text + "\n\n" + q}]}
t = time.time()
req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1200) as r:
    d = json.load(r)
dt = time.time() - t
ans = d["choices"][0]["message"].get("content") or ""
hits = [exp for _, (_, _, exp) in needles.items() if exp.lower() in ans.lower()]
print(f"prompt {d['usage']['prompt_tokens']} Token, {dt:.0f}s, Treffer {len(hits)}/3")
print("Antwort:", ans.strip().replace("\n", " | ")[:400])
