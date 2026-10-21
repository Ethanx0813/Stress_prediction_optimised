import numpy as np, torch, torch.nn as nn, json, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
torch.set_num_threads(8)
d=np.load('data/cache.npz');X,y=d['X'].astype(np.float64),d['y']
idx=np.arange(5000)
tr,tmp=train_test_split(idx,test_size=.30,random_state=42)   # same split as original notebook
va,te=train_test_split(tmp,test_size=.50,random_state=42)
def feats(X): return np.concatenate([1/X, np.log(X)],1)
F=feats(X); keep=F[tr].std(0)>1e-9
mu,sd=F[tr][:,keep].mean(0),F[tr][:,keep].std(0)
Fs=((F[:,keep]-mu)/sd).astype(np.float32)
ly=np.log(y); ym,ys=ly[tr].mean(),ly[tr].std(); yn=((ly-ym)/ys).astype(np.float32)
T=lambda a:torch.tensor(a)
class DNN(nn.Module):
    def __init__(s,i,h=512,p=0.1):
        super().__init__()
        s.i=nn.Linear(i,h)
        s.b=nn.ModuleList([nn.Sequential(nn.LayerNorm(h),nn.Linear(h,h),nn.GELU(),nn.Dropout(p),nn.Linear(h,h)) for _ in range(3)])
        s.o=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,1))
    def forward(s,x):
        x=s.i(x)
        for b in s.b:x=x+b(x)
        return s.o(x).squeeze(1)
def fit(seed,epochs=150):
    torch.manual_seed(seed)
    m=DNN(Fs.shape[1]);opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=1e-2)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=epochs*((len(tr)+63)//64))
    Xt,yt,Xv,yv=T(Fs[tr]),T(yn[tr]),T(Fs[va]),T(yn[va]);best=(9e9,None);hist=[]
    for e in range(epochs):
        m.train();perm=torch.randperm(len(tr));L=0
        for i in range(0,len(tr),64):
            b=perm[i:i+64];opt.zero_grad();l=nn.functional.smooth_l1_loss(m(Xt[b]+0.02*torch.randn(len(b),Xt.shape[1])),yt[b],beta=0.5);l.backward();opt.step();sch.step();L+=l.item()*len(b)
        m.eval()
        with torch.no_grad():v=nn.functional.mse_loss(m(Xv),yv).item()
        hist.append((L/len(tr),v))
        if v<best[0]:best=(v,{k:t.clone() for k,t in m.state_dict().items()})
    m.load_state_dict(best[1]);return m,hist
def pred(m,ix):
    m.eval()
    with torch.no_grad():return np.exp(m(T(Fs[ix])).numpy()*ys+ym)
if __name__=='__main__':
    ms=[];H=[]
    for s in range(5):
        m,h=fit(s);ms.append(m);H.append(h);p=pred(m,te);print('seed',s,'val-best test R2',round(r2_score(y[te],p),4),flush=True)
    p=np.mean([pred(m,te) for m in ms],0)
    print('ENSEMBLE R2',r2_score(y[te],p),'RMSE',mean_squared_error(y[te],p)**.5,'MAE',mean_absolute_error(y[te],p))
    p10,p25,p50=np.percentile(y[tr],[10,25,50])
    lab=lambda v:np.digitize(v,[p10,p25,p50],right=True)
    print('Bin acc',accuracy_score(lab(y[te]),lab(p)))
    import os; pass
    np.savez('data/results/dnn_out.npz',pred=p,true=y[te],hist=np.array(H))
    torch.save([m.state_dict() for m in ms],'data/models/dnn_ensemble.pth')
