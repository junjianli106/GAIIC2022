#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
import json
import re
import os
import pickle
from itertools import chain


# In[2]:


DIR = 'data/contest_data'
out_dir = 'data/tmp_data'


# In[3]:


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# # Replace

# 关键属性中存在等价的属性， 将它们替换为相同的字段

# In[4]:


with open(f'{DIR}/attr_to_attrvals.json', 'rb') as f:
    key_attr = json.load(f)


# In[5]:


labels = list(key_attr.keys())
labels


# In[6]:


def create_dict(x):
    out = {}
    for i in range(len(x)):
        out[x[i]] = x[0]
    return out


# In[7]:


attr = pd.Series(list(chain.from_iterable(list(key_attr.values()))))
replace = attr.str.split('=')
replace = replace.apply(create_dict)

dict_all = {}
for dicts in list(replace):
    dict_all.update(dicts)

dict_all


# In[8]:


def load_attr_dict():
    attr_dict = {}
    for attr, attrval_list in key_attr.items():
        attrval_list = list(map(lambda x: x.split('='), attrval_list))
        attrval_list = list(map(lambda x: x[0], attrval_list))
        attr_dict[attr] = attrval_list
    return attr_dict

attr_dict = load_attr_dict()
attr_dict # dict without duplicate


# # 辅助函数

# In[9]:


def create_DataFrame(path):
    df = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            df.append(list(json.loads(line).values()))
    return pd.DataFrame(df)

def replace_synonym(x):
    for key, value in dict_all.items():
        x = x.replace(key, value)
    return x


# # 处理没有关键字的数据

# In[10]:


coarse = create_DataFrame(f'{DIR}/train_coarse.txt')
coarse = coarse.drop(columns=2)
coarse.columns = ['img_name', 'title', '图文','feature']
coarse['图文'] = coarse['图文'].apply(lambda x:x['图文'])
# display(coarse.shape)
coarse.head()


# # 处理有关键字的部分

# In[11]:


fine = create_DataFrame(f'{DIR}/train_fine.txt')
fine = fine.drop(columns=2)
fine.columns = ['img_name', 'title', '图文','feature']
fine['图文'] = fine['图文'].apply(lambda x:x['图文'])
# display(fine.shape)
fine.head()


# # 图文匹配数据

# In[12]:


whole = pd.concat((fine, coarse)).reset_index(drop=True)
whole.columns = ['img_name', 'text', 'label','feature']
whole['text'] = whole['text'].agg(replace_synonym)
whole.to_feather(f'{out_dir}/whole.feather')
print(whole.shape)
whole.head()


# # 关键词数据

# ## 提取关键属性标签

# In[13]:


def extract_key_attr(x, label):
    ret = re.findall('|'.join(attr_dict[label]), x)
    if ret:
        if ret[0] in ['松紧', '拉链', '系带']:
            if '裤' in x and label == '裤门襟':
                return  ret[0]
            elif ('鞋' in x or '靴' in x) and label == '闭合方式':
                return  ret[0]
            return -1
        return ret[0]
    else:
        return -1
    
def get_type(x):
    for key, value in attr_dict.items():
        if x in value:
            return key


# ## 二分类

# ## 多分类

# In[14]:


multiclass = whole[whole.label==1].reset_index(drop=True)
multiclass = multiclass.rename(columns={'label':'图文'})
for label in labels:
    multiclass[label] = multiclass['text'].apply(extract_key_attr, label=label)
    multiclass.loc[multiclass[label]!=-1, label] = 1
# display(multiclass)
multiclass.to_feather(f'{out_dir}/multitask.feather')


# In[15]:


# for label in labels:
#     display(multiclass[label].value_counts())

