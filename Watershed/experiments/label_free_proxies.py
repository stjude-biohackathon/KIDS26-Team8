"""
Negative-result experiment: can a LABEL-FREE score pick the segmentation settings? We tested five (fraction of detected
objects / object voxels that pass the size+shape screen, spread of accepted sizes, accepted count, mask fraction) against
F1 over 16 settings on two corrected chunks. Rank correlations with F1 were +0.3 to +0.9 but none reliably picked the best
setting; letting the screen-pass signal choose would have put PFC below its validated F1. See docs/AUTO_CONFIG.md.

Edit CH below to point at your corrected chunks (raw, gt, first scored z, last scored z), then:
    python experiments/label_free_proxies.py
"""
import json
import numpy as np, tifffile
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr
from nuclei_stage1.config import Stage1Config
from nuclei_stage1.segmentation import run_stage1
from nuclei_stage1.evaluate import evaluate_one
U=''   # run from the folder holding the corrected chunks
CH={"hippo": ("gt_chunk3_raw.tif", "gt_chunk3.tif", 0, 128), "pfc": ("gt_chunk1_raw.tif", "gt_chunk1.tif", 1, 127)}
grid=[("raw",0,0,p,d) for p in (90,94) for d in (12,17)]+[("dog",lo,8,p,d) for lo in (1,2) for p in (92,94,96) for d in (12,17)]
def job(a):
    n,g=a; r,gtp,z0,z1=CH[n]; raw=tifffile.imread(r); gt=tifffile.imread(gtp).astype(np.int32)
    kind,lo,hi,p,d=g
    cfg=replace(Stage1Config(),target_foreground_percentile=float(p),min_seed_distance=d,seed_landscape_smooth_um=1.0,
                **({"mask_bandpass_um":(float(lo),float(hi))} if kind=="dog" else {}))
    acc,diff,lab=run_stage1(raw,cfg)
    m=evaluate_one(acc[z0:z1],gt[z0:z1],0.25,0.15)
    nl=len(np.unique(lab))-1; na=len(np.unique(acc))-1
    sz=np.bincount(acc.ravel())[1:]; sz=sz[sz>0]
    px=dict(acc_obj_frac=na/max(nl,1), acc_vox_frac=float((acc>0).sum()/max((lab>0).sum(),1)),
            size_cv=float(sz.std()/sz.mean()) if len(sz)>1 else 9, n_acc=na,
            mask_frac=float((lab>0).mean()))
    return n,g,m['f1'],px
if __name__=="__main__":
    jobs=[(n,g) for n in CH for g in grid]
    with ProcessPoolExecutor(2) as ex: res=list(ex.map(job,jobs))
    json.dump([(n,list(map(str,g)),f,px) for n,g,f,px in res],open('proxy_results.json','w'))
    for key in ("acc_obj_frac","acc_vox_frac","size_cv","n_acc","mask_frac"):
        out=[]
        for n in CH:
            x=[px[key] for nn,g,f,px in res if nn==n]; y=[f for nn,g,f,px in res if nn==n]
            out.append(f"{n}: rho={spearmanr(x,y)[0]:+.2f}")
        print(f"{key:14s}"," ".join(out))
    for n in CH:
        print(n,"best by F1:",max([(f,g) for nn,g,f,px in res if nn==n]))
        for k in ("acc_vox_frac","acc_obj_frac"):
            b=max([(px[k],g,f) for nn,g,f,px in res if nn==n]); print("   best by",k,b)
