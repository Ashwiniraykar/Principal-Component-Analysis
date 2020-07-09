
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[3]:


cancer.keys()


# In[4]:


print(cancer['DESCR'])


# In[6]:


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head(10)


# In[7]:


cancer['target'][0:25]


# In[8]:


cancer['target_names']


# In[9]:


C = np.dot(df, df.T)
v, w = np.linalg.eig(C)
print('Eigen-values: v=\n', v,'\n')
print('Eigen-vectors: w=\n', w)


# In[10]:


U, s, VT = np.linalg.svd(df)
print('Eigen-vectors: U=\n', U, U.shape,'\n')
print('Eigen-values: s=', s, s.shape)
print('Eigen-vectors: VT=', VT.shape,'\n')


# In[11]:


Ur = -U[:, 0:2]
print('Ur=\n', Ur)
wr = w[:, 0:2]
print('wr=\n', wr)
np.allclose(Ur, wr)


# In[12]:


Xr = np.dot(df.T, Ur)
plt.plot(Xr[:,0], Xr[:,1], 'bo')
plt.show()


# In[13]:


sv = np.cumsum(s)/sum(s)
plt.step(list(range(len(sv))), sv)
plt.show()


# In[14]:


np.round(np.dot(U.T, U), 0)


# In[15]:


np.round(np.dot(VT.T, VT),0)


# In[16]:


Sigma = np.zeros((df.shape[0], df.shape[1]))
Sigma[:df.shape[1], :df.shape[0]] = np.diag(s)
Xa = U.dot(Sigma.dot(VT))
print(Xa.shape)
res = np.allclose(df, Xa)
print('Is df equal to Xa? ', res)
df1 = pd.DataFrame(df.T)
df1.corr()


# In[17]:


plt.matshow(df1.corr(), cmap="OrRd")
plt.colorbar()
plt.show()


# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(df)
sc_data = sc.transform(df)


# In[19]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(sc_data)
xpca = pca.transform(sc_data)
sc_data.shape, xpca.shape


# In[20]:


plt.figure(figsize=(8,6))
plt.scatter(xpca[:,0], xpca[:,1], c=cancer['target'], cmap='plasma')
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.show()


# In[21]:


pca.components_


# In[22]:


df_components = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
df_components
plt.figure(figsize=(12,6))
sns.heatmap(df_components, cmap='plasma')
plt.show()

