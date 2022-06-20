#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import pickle
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling,PreTrainedTokenizer,PreTrainedTokenizerBase
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer,AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup
from transformers.activations import gelu

from models.nezha.configuration_nezha import NeZhaConfig
from models.nezha.modeling_nezha import NeZhaForMaskedLM
from tqdm import tqdm
import json
import numpy as np

import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
import json
import pickle
from itertools import chain

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# In[2]:


DIR = 'data/contest_data'


# In[3]:


with open(f'{DIR}/attr_to_attrvals.json', 'rb') as f:
    key_attr = json.load(f)


# In[4]:


labels = list(key_attr.keys()) 
labels


# In[5]:


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


# In[8]:


def replace_synonym(text, replace_dict):
    for key in replace_dict.keys():
        if key in text:
            text.replace(key, replace_dict[key])
    return text

def read_file(file_path):
    title_all = []
    with open(file_path, 'r') as f:
        for i, data in enumerate(tqdm(f)):
            data = json.loads(data)
            text = replace_synonym(data['title'], dict_all)
            title_all.append(text)
    return title_all


# In[9]:


fine = read_file('data/contest_data/train_fine.txt')
coarse = read_file('data/contest_data/train_coarse.txt')
testa = read_file('data/contest_data/preliminary_testA.txt')


# In[10]:


all_title = fine + coarse + testa


# In[11]:


len(all_title)


# In[12]:


with open('data/tmp_data/mlm_title_all.txt', 'w') as f:
    for i in range(len(all_title)):
        f.write(all_title[i]+"\n")
        
with open('data/tmp_data/testa.txt', 'w') as f:
    for i in range(len(testa)):
        f.write(testa[i]+"\n")


# In[14]:


model_path = 'data/pretrain_model/chinese-roberta-wwm-ext'


# In[15]:


tokenizer = BertTokenizer.from_pretrained(model_path)

#model = NeZhaForMaskedLM.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
print('num of parameters: ', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[16]:


dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/tmp_data/mlm_title_all.txt",  # mention train text file here
    block_size=40)

test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/tmp_data/testa.txt",  # mention valid text file here
    block_size=40)


# In[17]:


print('num of lines: ', len(dataset)) # No of lines in your datset

training_args = TrainingArguments(
    output_dir='data/pretrain_model/chinese-roberta-wwm-ext',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    num_train_epochs=20,
    per_device_train_batch_size=512,
    save_steps=2000,
    eval_steps=2000,
    fp16=True
)


# In[18]:


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=test_dataset
)


# In[19]:


trainer.train()
trainer.save_model('data/pretrain_model/chinese-roberta-wwm-ext')


# In[ ]:




