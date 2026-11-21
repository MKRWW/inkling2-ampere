"""KL divergence, top-1 agreement and perplexity of two OpenAI-compatible vLLM servers (prompt_logprobs).

Usage:
  kld_eval.py collect <server-url> corpus.json ref.json    # builds corpus.json on first run
  kld_eval.py collect <other-url>  corpus.json test.json
  kld_eval.py compare ref.json test.json
"""
import json, sys, random, glob, gzip, os, math, urllib.request, time
MODEL="inkling-2-mini"; K=20
def post(url, body):
    r=urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))

DE_GLOB=os.environ.get("CORPUS_DE","")   # e.g. "/texts/de/**/*.md" (German prose)
EN_GLOB=os.environ.get("CORPUS_EN","/usr/share/doc/*/README*")
CODE_GLOB=os.environ.get("CORPUS_CODE","/usr/lib/python3*/*.py")
LONG_DE=os.environ.get("LONG_DE","")
LONG_EN=os.environ.get("LONG_EN","/usr/share/common-licenses/GPL-3")

def build_corpus(url):
    rnd=random.Random(1234)
    def chunks(files, n, kind):
        out=[]; rnd.shuffle(files)
        for fn in files:
            try:
                t=gzip.open(fn,"rt",errors="ignore").read() if fn.endswith(".gz") else open(fn,errors="ignore").read()
            except Exception: continue
            if len(t)<2000: continue
            st=rnd.randrange(0, len(t)-1600); out.append((kind, t[st:st+1600]))
            if len(out)>=n: break
        return out
    de=chunks([f for f in glob.glob(DE_GLOB,recursive=True) if os.path.getsize(f)>4000], 70, "de") if DE_GLOB else []
    en=chunks(glob.glob(EN_GLOB,recursive=True), 70, "en")
    code=chunks(glob.glob(CODE_GLOB,recursive=True), 60, "code")
    items=[]
    for kind,t in de+en+code:
        ids=post(url+"/tokenize",{"model":MODEL,"prompt":t,"add_special_tokens":False})["tokens"][:320]
        items.append({"kind":kind,"ids":ids})
    # long texts for perplexity
    for kind,fn in (("long_de",LONG_DE),("long_en",LONG_EN)):
        if not fn: continue
        ids=post(url+"/tokenize",{"model":MODEL,"prompt":open(fn,errors="ignore").read(),"add_special_tokens":False})["tokens"][:6000]
        items.append({"kind":kind,"ids":ids,"src":fn})
    return items

def collect(url, corpus_path, out):
    if os.path.exists(corpus_path): items=json.load(open(corpus_path))
    else:
        items=build_corpus(url); json.dump(items,open(corpus_path,"w"))
    res=[]; t0=time.time()
    for i,it in enumerate(items):
        r=post(url+"/v1/completions",{"model":MODEL,"prompt":it["ids"],"max_tokens":1,"temperature":0,"prompt_logprobs":K})
        pl=r["choices"][0]["prompt_logprobs"]
        pos=[]
        for j in range(1,len(it["ids"])):
            d=pl[j]; tok=str(it["ids"][j])
            top={k:v["logprob"] for k,v in d.items() if v["rank"]<=K}
            pos.append([top, d[tok]["logprob"]])
        res.append({"kind":it["kind"],"pos":pos})
        if i%20==0: print(i,len(items),f"{time.time()-t0:.0f}s",flush=True)
    json.dump(res,open(out,"w"))

def compare(a,b):
    A=json.load(open(a)); B=json.load(open(b))
    agg={}
    for ra,rb in zip(A,B):
        kind=ra["kind"]; s=agg.setdefault(kind,{"n":0,"kl":0,"top1":0,"nlla":0,"nllb":0,"kls":[]})
        for (ta,la),(tb,lb) in zip(ra["pos"],rb["pos"]):
            mb=min(tb.values())
            kl=0; pa=0; qb=0
            for t,l in ta.items():
                p=math.exp(l); q=tb.get(t,mb); pa+=p; qb+=math.exp(q); kl+=p*(l-q)
            ra_t=max(1-pa,1e-12); rb_t=max(1-qb,1e-12); kl+=ra_t*math.log(ra_t/rb_t)
            kl=max(kl,0)
            s["n"]+=1; s["kl"]+=kl; s["kls"].append(kl)
            s["top1"]+= max(ta,key=ta.get)==max(tb,key=tb.get)
            s["nlla"]-=la; s["nllb"]-=lb
    tot={"n":0,"kl":0,"top1":0,"nlla":0,"nllb":0,"kls":[]}
    def line(k,s):
        ks=sorted(s["kls"]); 
        print(f"{k:8s} n={s['n']:6d}  KLD mean={s['kl']/s['n']:.5f} p99={ks[int(.99*len(ks))]:.4f} max={ks[-1]:.3f}  top1={100*s['top1']/s['n']:.2f}%  PPL A={math.exp(s['nlla']/s['n']):.4f} B={math.exp(s['nllb']/s['n']):.4f}")
    for k in sorted(agg):
        line(k,agg[k])
        if not k.startswith("long"):
            for f in tot: tot[f]+=agg[k][f]
    line("mixed",tot)

if __name__=="__main__":
    if sys.argv[1]=="collect": collect(sys.argv[2],sys.argv[3],sys.argv[4])
    else: compare(sys.argv[2],sys.argv[3])
