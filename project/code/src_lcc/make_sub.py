#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pandas as pd


# In[2]:


DIR = '../../data/contest_data/'
data_dir = '../../data/tmp_data'
out_dir = '../../data/submission'
out_file =  '../../data/submission/results_lcc.txt'


# In[3]:


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# In[4]:


if os.path.exists(out_file):
    os.remove(out_file)


# In[5]:


test = pd.read_table(DIR + 'preliminary_testB.txt', header=None)
test = test[0].apply(eval)


# In[6]:


test


# In[7]:


whole = pd.read_csv(f'{data_dir}/whole_pred.csv')
whole.index = whole['img_name']
whole = whole['pred']


# In[8]:


detail = pd.read_csv(f'{data_dir}/detail_pred.csv')
detail.index = detail[['img_name','type']]
detail = detail['pred']


# In[9]:


def get_label(x, whole, detail):
    labels = {}
    match = False
    labels['图文'] = whole[x['img_name']].tolist()
#     if labels['图文'] == 1:
#         match = True
    x['query'].remove('图文')
    for key in x['query']:   
        if match:
            labels[key] = 1    
        else:
            labels[key] = detail[(x['img_name'], key)].tolist()
#             if labels[key] == 0:
#                 labels['图文'] = 0
    return labels

def add_img_name(x):
    labels = {}
    labels['img_name'] = x['img_name']
    labels['match'] = x['match']
    
    return labels


# In[10]:


sub = pd.DataFrame([])
sub['img_name'] = test.apply(lambda x:x['img_name'])
query = pd.DataFrame([])
query['query'] = test.apply(lambda x:x['query'])
query['img_name'] = sub['img_name'].copy()
sub['match'] = query.apply(get_label, whole=whole,detail=detail, axis=1)
sub = sub.apply(add_img_name, axis=1)


# In[11]:


for i in sub.index:
    with open(out_file, 'a+', encoding='utf-8') as f:
        line = json.dumps(sub[i])
        f.write(line +'\n')


# In[12]:


sub


# In[13]:


(sub.apply(lambda x:x['match']['图文'] == 0 and 1 in x['match'].values())).sum()


# In[14]:


(sub.apply(lambda x:x['match']['图文'] == 1 and 0 in x['match'].values())).sum()

