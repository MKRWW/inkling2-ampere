"""Requantize the MXFP8 experts of layers 20-23 to NVFP4, in exactly the layout of the other layers.

Usage: nvfp4_convert.py <source checkpoint dir> <new dir>   (CPU only, ~2 min, source stays untouched)
Runs a self-test first: requantizing existing NVFP4 experts must reproduce them bit for bit.
"""
import torch, json, struct, os, sys, shutil, time
SRC=sys.argv[1] if len(sys.argv)>1 else "/src"; DST=sys.argv[2] if len(sys.argv)>2 else "/dst"
FN="model-00001-of-00001.safetensors"
LAYERS=(20,21,22,23); DEN=2688.0
LUT=torch.tensor([0,0.5,1,1.5,2,3,4,6,-0.0,-0.5,-1,-1.5,-2,-3,-4,-6])

def dec_fp4(u8):
    return torch.stack([LUT[(u8&0xF).long()],LUT[(u8>>4).long()]],-1).flatten(-2)

def e2m1(x):
    a=x.abs()
    code=torch.zeros_like(a,dtype=torch.uint8)
    for thr,c,le in ((0.25,1,True),(0.75,2,False),(1.25,3,True),(1.75,4,False),(2.5,5,True),(3.5,6,False),(5.0,7,True)):
        code=torch.where(a>thr if le else a>=thr, torch.tensor(c,dtype=torch.uint8), code)
    return code | ((x<0)|((x==0)&torch.signbit(x))).to(torch.uint8)<<3

def quant_expert(W):  # W float32 [N,K] -> packed u8 [N,K/2], scale e4m3 [N,K/16], amax
    amax=W.abs().max()
    g=amax/DEN
    B=W.view(W.shape[0],-1,16)
    bm=B.abs().amax(-1)
    s=(bm/(6.0*g)).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf=s.float()
    denom=(sf*g).unsqueeze(-1)
    q=torch.where(denom>0, B/denom.clamp(min=1e-30), torch.zeros_like(B))
    c=e2m1(q).view(W.shape[0],-1)
    packed=(c[:,0::2] | (c[:,1::2]<<4)).contiguous()
    return packed, s, amax

def deq_nvfp4(packed,s,amax):
    v=dec_fp4(packed)
    return (v.view(v.shape[0],-1,16)*s.float().unsqueeze(-1)*(amax/DEN)).view(v.shape[0],-1)

def read_header(path):
    f=open(path,"rb"); n=struct.unpack("<Q",f.read(8))[0]; h=json.loads(f.read(n)); return f,h,8+n

f,h,base=read_header(f"{SRC}/{FN}")
def raw(name):
    a,b=h[name]["data_offsets"]; f.seek(base+a); return f.read(b-a)
def tensor(name):
    from safetensors import safe_open
    return SO.get_tensor(name)
from safetensors import safe_open
SO=safe_open(f"{SRC}/{FN}","pt")

# --- self-test: requantizing an existing NVFP4 expert must reproduce it bit-exactly
for L,w in ((0,"w13"),(19,"w2")):
    p=f"model.llm.layers.{L}.mlp.experts.{w}_weight"
    am=SO.get_tensor(p+"_amax")
    for e in (0,77):
        q0=SO.get_slice(p)[e:e+1][0]; s0=SO.get_slice(p+"_scale")[e:e+1][0]
        W=deq_nvfp4(q0,s0,am[e])
        q1,s1,a1=quant_expert(W)
        print("selftest",L,w,e,"bytes equal:",bool((q1==q0).all()),"scales equal:",bool((s1.view(torch.uint8)==s0.view(torch.uint8)).all()),"amax",a1.item(),am[e].item(),flush=True)

# --- new header
meta=h.pop("__metadata__",None)
order=sorted(h.keys(), key=lambda k:h[k]["data_offsets"][0])
newspec={}  # name -> (dtype, shape)
for k in order:
    newspec[k]=(h[k]["dtype"],h[k]["shape"])
for L in LAYERS:
    for w in ("w13","w2"):
        p=f"model.llm.layers.{L}.mlp.experts.{w}_weight"
        E,N,K=h[p]["shape"]
        newspec[p]=("U8",[E,N,K//2]); newspec[p+"_scale"]=("F8_E4M3",[E,N,K//16])
        newspec[p+"_amax"]=("F32",[E]); newspec[p+"_outer_scale_denominator"]=("F32",[1])
names=list(newspec.keys())
size={"U8":1,"F8_E4M3":1,"F32":4,"BF16":2}
hdr={} ; off=0
for k in names:
    dt,sh=newspec[k]; n=size[dt]
    for d in sh: n*=d
    hdr[k]={"dtype":dt,"shape":sh,"data_offsets":[off,off+n]}; off+=n
if meta: hdr["__metadata__"]=meta
hj=json.dumps(hdr,separators=(",",":")).encode()
hj+=b" "*((8-len(hj)%8)%8)

converted={}
stats=[]
def convert_layer(L):
    t0=time.time()
    for w in ("w13","w2"):
        p=f"model.llm.layers.{L}.mlp.experts.{w}_weight"
        E=h[p]["shape"][0]
        Q=[];S=[];A=[]; err=0.0; ref=0.0
        for e in range(E):
            q8=SO.get_slice(p)[e:e+1][0].float(); sc=SO.get_slice(p+"_scale")[e:e+1][0]
            W=(q8.view(q8.shape[0],-1,32)*torch.pow(2.0,sc.float()-127).unsqueeze(-1)).view(q8.shape[0],-1)
            pk,s,a=quant_expert(W)
            R=deq_nvfp4(pk,s,a)
            err+=((R-W)**2).sum().item(); ref+=(W**2).sum().item()
            Q.append(pk);S.append(s);A.append(a)
        converted[p]=torch.stack(Q); converted[p+"_scale"]=torch.stack(S)
        converted[p+"_amax"]=torch.stack(A).float(); converted[p+"_outer_scale_denominator"]=torch.tensor([DEN])
        rel=(err/ref)**0.5
        stats.append((L,w,rel)); print(f"layer {L} {w}: rel RMSE vs MXFP8 = {rel:.4f}  ({time.time()-t0:.0f}s)",flush=True)

def tobytes(t):
    if t.dtype==torch.float8_e4m3fn: t=t.view(torch.uint8)
    return t.contiguous().numpy().tobytes()

os.makedirs(DST,exist_ok=True)
tmp=f"{DST}/{FN}.part"
with open(tmp,"wb") as o:
    o.write(struct.pack("<Q",len(hj))); o.write(hj)
    for k in names:
        m=k.split(".")
        if k in h and not any(k.startswith(f"model.llm.layers.{L}.mlp.experts.") for L in LAYERS):
            o.write(raw(k)); continue
        L=int(m[3])
        if k not in converted: convert_layer(L)
        b=tobytes(converted.pop(k)); assert len(b)==hdr[k]["data_offsets"][1]-hdr[k]["data_offsets"][0],k
        o.write(b)
os.rename(tmp,f"{DST}/{FN}")

# --- index + quant config + other files
for fn in os.listdir(SRC):
    if fn in (FN,"model.safetensors.index.json","hf_quant_config.json"): continue
    s=f"{SRC}/{fn}"; d=f"{DST}/{fn}"
    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s,d,dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy2(s,d)
idx=json.load(open(f"{SRC}/model.safetensors.index.json"))
for k in names: idx["weight_map"].setdefault(k,FN)
idx.setdefault("metadata",{})["total_size"]=off
json.dump(idx,open(f"{DST}/model.safetensors.index.json","w"),indent=2)
qc=json.load(open(f"{SRC}/hf_quant_config.json"))
groups=qc["model"]["parameter_groups"]
mx=[g for g in groups if g["compute_dtype"]=="mxfp8"][0]; nv=[g for g in groups if g["compute_dtype"]=="nvfp4"][0]
nv["tensors"]=sorted(nv["tensors"]+mx["tensors"]); groups.remove(mx)
json.dump(qc,open(f"{DST}/hf_quant_config.json","w"),indent=2)
print("old size",os.path.getsize(f"{SRC}/{FN}"),"new size",os.path.getsize(f"{DST}/{FN}"))
print("STATS",stats)
