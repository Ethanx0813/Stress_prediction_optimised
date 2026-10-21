import zipfile, io, numpy as np, pandas as pd
z = zipfile.ZipFile('data/DL_data.zip')
names = z.namelist()
X = pd.read_excel(io.BytesIO(z.read('prob1_data/output.xlsx'))).values.astype(np.float32)
sf = sorted([n for n in names if 'prob1_data/stress/' in n and n.endswith('.txt')], key=lambda x:int(x.split('_')[-1].replace('.txt','')))
y = np.array([np.loadtxt(io.StringIO(z.read(s).decode()))[:,0].max() for s in sf], dtype=np.float64)
np.savez('data/cache.npz', X=X, y=y)
print(X.shape, y.shape, y.min(), y.max(), np.percentile(y,[10,25,50,75,90]))
print(X[:3,:8]); print(X.min(0)[:5], X.max(0)[:5], X.std(0)[:10])
