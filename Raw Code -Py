
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib import*

from sklearn.decomposition import PCA 


# In[50]:


data = pd.read_csv('gene_data-.csv')
meta = pd.read_csv('Meta-data-sheet.csv')


# In[51]:


meta.head()


# In[3]:


data.head()


# In[4]:


len(data['symbol'].unique())


# In[5]:


len(data)


# In[6]:


dta_ch = data.iloc[:,2:]


# In[7]:


dta_ch=dta_ch.replace({'ssssss': '39.0', 'hhhh' : '321.43'}, regex=True)


# In[8]:


dta_ch.head()


# In[31]:


X=dta_ch.values
X=X.astype(float)


# In[32]:


X[np.isnan(X)] = np.median(X[~np.isnan(X)])


# In[33]:


X.shape


# In[34]:


X = X.T
X.shape


# In[35]:


X_std = StandardScaler().fit_transform(X)


# In[36]:


pca = PCA(n_components=2)
pca.fit(X_std)


# In[37]:


var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()


# In[38]:


y = pca.components_


# In[39]:


y.shape


# In[40]:


z = pca.explained_variance_


# In[41]:


a = 0.0
for i in range(len(z)):
    a = a+ float(z[i])
print a


# In[42]:


xn = pca.fit_transform(X)


# In[43]:


xn.shape


# In[52]:


target = np.array(meta['Time'])


# In[61]:


x_min, x_max = xn[:, 0].min() - 10000, xn[:, 0].max() + 10000
y_min, y_max = xn[:, 1].min() - 10000, xn[:, 1].max() + 10000


# In[62]:


plt.figure(figsize= (15,10))

plt.scatter(xn[:, 0], xn[:, 1], c=target, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()


# In[69]:


import seaborn as sns


# In[70]:


xx = list(xn[:, 0])
yy = list(xn[:, 1])


# In[72]:


d = {'PC1':xx , 'PC2': yy,'Target': target }
df = pd.DataFrame(data=d)
df.head()


# In[75]:


sns.pairplot(df, hue="Target")
plt.show()


# In[ ]:




