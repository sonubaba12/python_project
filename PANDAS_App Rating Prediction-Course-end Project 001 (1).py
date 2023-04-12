#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('googleplaystore.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().any()


# In[8]:


data.isnull().sum()


# In[9]:


data = data.dropna()
data.isnull().any()


# In[11]:


data.shape


# In[15]:


data["Size"] = [float(i.split('M')[0]) if isinstance(i, str) and 'M' in i else float(0) for i in data["Size"]]


# In[16]:


data.head()


# In[17]:


data["Size"] = 1000 * data["Size"]


# In[18]:


data


# In[19]:


data.info()


# In[20]:


data["Reviews"] = data["Reviews"].astype(float)


# In[21]:


data.info()


# In[22]:


data["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in data["Installs"] ]
data.head()


# In[23]:


data.info()


# In[24]:


data["Installs"] = data["Installs"].astype(int)


# In[25]:


data.info()


# In[26]:


data['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in data['Price'] ]


# In[28]:


data.head()


# In[29]:


data.info()


# In[30]:


data["Price"] = data["Price"].astype(int)


# In[31]:


data.info()


# In[32]:


data.shape


# In[33]:


data.drop(data[(data['Reviews'] < 1) & (data['Reviews'] > 5 )].index, inplace = True)
data.shape


# In[34]:


data.shape


# In[35]:


data.drop(data[data['Installs'] < data['Reviews'] ].index, inplace = True)
data.shape


# In[36]:


data.shape


# In[37]:


data.drop(data[(data['Type'] =='Free') & (data['Price'] > 0 )].index, inplace = True)


# In[38]:


data.shape


# In[39]:


sns.set(rc={'figure.figsize':(12,8)})
sns.boxplot(data['Price'])


# In[42]:


sns.boxplot(data['Reviews'])


# In[43]:


sns.boxplot(data['Rating'])


# In[45]:


sns.boxplot(data['Size'])


# In[46]:


more = data.apply(lambda x : True
            if x['Price'] > 200 else False, axis = 1) 


# In[47]:


more_count = len(more[more == True].index) 


# In[48]:


data.shape


# In[49]:


data.drop(data[data['Price'] > 200].index, inplace = True)


# In[50]:


data.shape


# In[51]:


data.drop(data[data['Reviews'] > 2000000].index, inplace = True)


# In[52]:


data.shape


# In[53]:


data.quantile([.1, .25, .5, .70, .90, .95, .99], axis = 0)


# In[54]:


# dropping more than 10000000 Installs value
data.drop(data[data['Installs'] > 10000000].index, inplace = True)


# In[55]:


data.shape


# In[56]:


sns.scatterplot(x='Rating',y='Price',data=data)


# Yes, Paid apps are higher ratings comapre to free apps.

# In[58]:


sns.scatterplot(x='Rating',y='Size',data=data)


# Yes it is clear that heavior apps are rated better.

# In[60]:


sns.scatterplot(x='Rating',y='Reviews',data=data)


# It is cristal clear that more reviews makes app rating better.

# In[62]:


sns.boxplot(x="Rating", y="Content Rating", data=data)


# pps which are for everyone has more bad ratings compare to other sections as it has so much outliers value, while 18+ apps have better ratings.

# In[64]:


sns.boxplot(x="Rating", y="Category", data=data)


# Events category has best ratings compare to others.

# In[67]:


inp1 = data


# In[68]:


inp1.head()


# In[69]:


inp1.skew()


# In[70]:


reviewskew = np.log1p(inp1['Reviews'])
inp1['Reviews'] = reviewskew


# In[71]:


reviewskew.skew()


# In[72]:


installsskew = np.log1p(inp1['Installs'])
inp1['Installs']


# In[73]:


installsskew.skew()


# In[74]:


inp1.head()


# In[75]:


inp1.drop(["Last Updated","Current Ver","Android Ver","App","Type"],axis=1,inplace=True)


# In[76]:


inp1.head()


# In[77]:


inp1.shape


# In[78]:


inp2 = inp1


# In[79]:


inp2.head()


# # Let's apply Dummy EnCoding on Column "Category"

# In[81]:


#unique values in Column "Category"
inp2.Category.unique()


# In[82]:


inp2.Category = pd.Categorical(inp2.Category)
x = inp2[['Category']]
del inp2['Category']
dummies = pd.get_dummies(x, prefix = 'Category')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[83]:


inp2.shape


# # apply Dummy EnCoding on Column "Genres"

# In[84]:


#get unique values in Column "Genres"
inp2["Genres"].unique()


# => Since, There are too many categories under Genres. Hence, we will try to reduce some categories which have very few samples under them and put them under one new common category i.e. "Other".

# In[85]:


lists = []
for i in inp2.Genres.value_counts().index:
    if inp2.Genres.value_counts()[i]<20:
        lists.append(i)
inp2.Genres = ['Other' if i in lists else i for i in inp2.Genres] 


# In[86]:


inp2["Genres"].unique()


# In[87]:


inp2.Genres = pd.Categorical(inp2['Genres'])
x = inp2[["Genres"]]
del inp2['Genres']
dummies = pd.get_dummies(x, prefix = 'Genres')
inp2 = pd.concat([inp2,dummies], axis=1)


# In[88]:


inp2.head()


# In[89]:


inp2.shape


# Let's apply Dummy EnCoding on Column "Content Rating"

# In[90]:


#get unique values in Column "Content Rating"
inp2["Content Rating"].unique()


# In[91]:


inp2['Content Rating'] = pd.Categorical(inp2['Content Rating'])

x = inp2[['Content Rating']]
del inp2['Content Rating']

dummies = pd.get_dummies(x, prefix = 'Content Rating')
inp2 = pd.concat([inp2,dummies], axis=1)
inp2.head()


# In[92]:


inp2.shape


# In[93]:


from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse


# In[94]:


d1 = inp2
X = d1.drop('Rating',axis=1)
y = d1['Rating']

Xtrain, Xtest, ytrain, ytest = tts(X,y, test_size=0.3, random_state=5)


# In[95]:


reg_all = LR()
reg_all.fit(Xtrain,ytrain)


# In[96]:


R2_train = round(reg_all.score(Xtrain,ytrain),3)
print("The R2 value of the Training Set is : {}".format(R2_train))


# In[97]:


R2_test = round(reg_all.score(Xtest,ytest),3)
print("The R2 value of the Testing Set is : {}".format(R2_test))


# In[ ]:




