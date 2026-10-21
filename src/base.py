import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
d=np.load('data/cache.npz');X,y=d['X'],d['y']
tr,te=train_test_split(np.arange(5000),test_size=.15,random_state=42)
for name,F in [('raw',X),('log',np.log(X)),('inv',1/X),('inv2',1/X**2)]:
    for t in ['y','logy']:
        yt=y if t=='y' else np.log(y)
        m=Ridge(alpha=1e-3 if name=='raw' else 1).fit((F[tr]-F[tr].mean(0))/F[tr].std(0),yt[tr])
        p=m.predict((F[te]-F[tr].mean(0))/F[tr].std(0))
        if t=='logy':p=np.exp(p)
        print(name,t,round(r2_score(y[te],p),4))
print(np.corrcoef(X.min(1),y)[0,1], np.corrcoef(np.log(X.min(1)),np.log(y))[0,1])
c=[abs(np.corrcoef(X[:,i],np.log(y))[0,1]) for i in range(226)]
print(np.round(c[:40],2)); print(np.argsort(c)[::-1][:15])
