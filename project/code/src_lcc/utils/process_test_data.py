#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
import json
import os
from itertools import chain
import argparse

parser = argparse.ArgumentParser(description='choose test file')
parser.add_argument('testFile', help='A or B')
args = parser.parse_args()
print(f'process test file {args.testFile}')

# In[2]:
DIR = 'data/contest_data'
out_dir = 'data/tmp_data'

# DIR = '../../../data/contest_data'
# out_dir = '../../../data/tmp_data'


# In[3]:


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# In[4]:


with open(f'{DIR}/attr_to_attrvals.json', 'rb') as f:
    key_attr = json.load(f)


# In[ ]:


labels = list(key_attr.keys()) 
labels


# In[ ]:


def create_dict(x):
    out = {}
    for i in range(len(x)):
        out[x[i]] = x[0]
    return out


# In[ ]:


attr = pd.Series(list(chain.from_iterable(list(key_attr.values()))))
replace = attr.str.split('=')
replace = replace.apply(create_dict)

dict_all = {}
for dicts in list(replace):
    dict_all.update(dicts)

dict_all


# In[ ]:


def load_attr_dict():
    attr_dict = {}
    for attr, attrval_list in key_attr.items():
        attrval_list = list(map(lambda x: x.split('='), attrval_list))
        attrval_list = list(map(lambda x: x[0], attrval_list))
        attr_dict[attr] = attrval_list
    return attr_dict

key_attr = load_attr_dict()
key_attr # dict without duplicate


# In[ ]:


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


# In[ ]:


def get_key(df):
    out = pd.DataFrame([])
    
    def process_label(x,label):
        if label in x['query']:
            types = list(key_attr[label])
            for Type in types:
                if Type in x['title']:
                    return Type
        else:
            return pd.NA
                    
    for label in labels:
        df[label] = df.apply(process_label,label=label, axis=1)
    for label in labels:
        add = df.dropna(subset=[label])[['img_name','feature', label, 'title']]
        add['type'] = label
        add = add.rename(columns={label:'text'})
        out = pd.concat((out, add))
    return out


# In[ ]:


test = create_DataFrame(f'{DIR}/preliminary_test{args.testFile}.txt')
test.columns = ['img_name', 'text', 'query', 'feature']
test['text'] = test['text'].agg(replace_synonym)
# display(test.head())
test.to_feather(f'{out_dir}/test_whole.feather')
test.columns = ['img_name', 'title', 'query', 'feature']
test = get_key(test).reset_index(drop=True)
test.to_feather(f'{out_dir}/test_detail.feather')
test.head()

