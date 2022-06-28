#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression


# In[7]:


data = pd.read_csv('nba_2020_advanced.csv')


# In[8]:


data.head()


# In[9]:


data


# In[10]:


VORP_DATA = data.sort_values(by=['VORP'])
df = VORP_DATA[-10:]
VORP = df['VORP']
Player = df['Player']
VORP


# In[11]:


MVPPTS = [26.0, 9.0, 168.0, 82.0, 200.0, 18.0, 23.0, 753.0, 962.0, 367.0]
MVPPOS = [7, 11, 5, 6, 4, 9, 8, 2, 1, 3]


# In[12]:


df['MVPPTS'] = MVPPTS
df['MVPPOS'] = MVPPOS
df


# In[19]:


fig, ax = plt.subplots()


ax.scatter(x=df['MVPPOS'], y=df['VORP'])
ax.set_xlabel('MVPPOS')
ax.set_ylabel('VORP')
ax.invert_xaxis()


for idx, row in df.iterrows():
    ax.annotate(row['Player'], (row['MVPPOS'], row['VORP']) )
    


# In[23]:


x= df['MVPPOS']
y= df['VORP']

x = np.array(x).reshape(-1,1)     
y = np.array(y).reshape(-1,1)    

model = LinearRegression()
model.fit(x, y)


# In[24]:


r2 = round(model.score(x,y), 2)            
predicted_y = model.predict(x)    


# In[37]:


fig, ax = plt.subplots()
ax.scatter(x, y, s=15, alpha=.5)                            
ax.plot(x, predicted_y, color = 'black')                            
ax.set_xlabel('MVP Ranking')                                  
ax.set_ylabel('VORP')    
ax.invert_xaxis()

for idx, row in df.iterrows():
    ax.annotate(row['Player'], (row['MVPPOS'], row['VORP']) )


# In[33]:


plt.text(10,25, f'R2={r2}')


# In[ ]:





# In[ ]:




