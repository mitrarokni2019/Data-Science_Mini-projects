import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
#df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/LAX-air-pollution.csv')
df=pd.read_csv('D:/Laptop backup 2/course work2/course works2/Data Science program/files/men-track-records.csv')
X=df.iloc[:,:-1]
#X=df
#scaledX=scale(X)
p=PCA()
p.fit(X)
W=p.components_.T
y=p.fit_transform(X)
yhat=X.dot(W)
plt.figure(1)
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="red",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

names=df.Country
names=names.agg(lambda x: x[:5])
for i, txt in enumerate(names):
    plt.annotate(txt, (yhat.iloc[i,0], yhat.iloc[i,1]))

pd.DataFrame(W[:,:3],index=df.columns[:-1],columns=['PC1','PC2','PC3'])

pd.DataFrame(p.explained_variance_ratio_,index=np.arange(8)+1,columns=['Explained Variability'])
plt.figure(2)
plt.bar(np.arange(1,9),p.explained_variance_,color="blue",edgecolor="Red")
