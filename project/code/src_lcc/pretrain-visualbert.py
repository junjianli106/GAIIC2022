#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import random
import pandas as pd


# # CFG

# In[2]:


class CFG:
    data_dir = 'data/tmp_data/'
    out_dir = 'data/pretrain_model'
    scheduler='cosine'
    train_file = 'whole.feather' # whole/detail/post_processing
    model_name = 'uclanlp/visualbert-vqa-coco-pre'
    
    seed = 42
    batch_size = 32
    dropout = 0.1 
    text_dim = 768
    img_dim = 2048
    
    transformer_lr = 2e-5
    clf_lr = 1e-4
    weight_decay = 0.01
    eps=1e-6
    betas=(0.9, 0.999)
    num_warmup_steps=0
    max_norm = 1000
    num_cycles=0.5
    
    epochs=40
    
    
    patience = 5


# In[3]:


if not os.path.exists(CFG.out_dir):
    os.makedirs(CFG.out_dir)


# # Set Seed

# In[4]:


import numpy as np
import torch
import os
import random

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CFG.seed)


# # Get Model

# In[5]:


from transformers import AutoTokenizer


# In[6]:


CFG.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


# # Read In Data

# In[7]:


df = pd.read_feather(CFG.data_dir + CFG.train_file)
df = df[df.label==1].reset_index(drop=True)
# display(df)


# In[8]:


vocab = list(set(''.join(df['text'].values)))


# # Dataset

# In[9]:


from torch.utils.data import DataLoader, Dataset


# In[10]:


def mask(x):
    for ch in x:
        if random.random() > 0.85:
            prob = random.random()
            if prob < 0.8:
                x = x.replace(ch, '{mask}')
            elif 0.9 > prob >= 0.8:
                x = x.replace(ch, random.choice(vocab))
    return x 


# In[11]:


class MyDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.text = df['text']
        self.feature = df['feature']
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        label = self.text[index]
        text = mask(label)
        return text, torch.tensor(self.feature[index]).float(), label


# # Helper Function

# In[12]:


from torch.optim import AdamW, Adam
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


# In[13]:


def get_optimizer(model, CFG):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {'params': [p for n, p in model.visual_bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': CFG.transformer_lr, 'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in model.visual_bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': CFG.transformer_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "visual_bert" not in n and not any(nd in n for nd in no_decay)],
             'lr': CFG.clf_lr, 'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in model.named_parameters() if "visual_bert" not in n and any(nd in n for nd in no_decay)],
             'lr': CFG.clf_lr, 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_parameters, lr=CFG.transformer_lr, eps=CFG.eps, betas=CFG.betas)
    return optimizer

def get_scheduler(CFG, optimizer, num_train_steps):
    if CFG.scheduler=='linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif CFG.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=CFG.num_cycles
        )
    return scheduler


# # Train

# In[14]:


from tqdm.auto import tqdm
import torch.nn as nn
from transformers import VisualBertForPreTraining
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# In[15]:


dataset = MyDataset(df)
loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)

model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre").cuda()

optimizer = get_optimizer(model, CFG)
scheduler = get_scheduler(CFG, optimizer, int(len(df) / CFG.batch_size * CFG.epochs))

best_score = float(np.Inf)
for epoch in range(CFG.epochs):
    dataset_size = 0
    running_loss = 0
    model.train()
    bar = tqdm(loader, total=len(loader))
    for text, feature,label in bar:
        text = CFG.tokenizer(
            text, return_tensors='pt', add_special_tokens=True, padding=True)
        for k, v in text.items():
            text[k] = v.cuda()
        feature = feature.unsqueeze(1).cuda()
        text.update(
            {
                'visual_embeds': feature,
                'visual_attention_mask': torch.ones(feature.shape[:-1], dtype=torch.float).cuda(),
                "visual_token_type_ids":torch.ones(feature.shape[:-1], dtype=torch.long).cuda(),
            }
        )
        max_length = text["input_ids"].shape[-1] + feature.shape[-2]
        
        label = CFG.tokenizer(
            label, return_tensors='pt', add_special_tokens=True, padding="max_length", max_length=max_length)['input_ids'].cuda()
        
        
        optimizer.zero_grad()
        loss = model(**text, labels=label).loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
        optimizer.step()
        scheduler.step()

        dataset_size += label.shape[0]
        running_loss += loss.item() * label.shape[0]
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    
    if epoch_loss < best_score:
        print(f'best_score improved from {best_score} -----> {epoch_loss}')
        best_score = epoch_loss
        torch.save(model.visual_bert.state_dict(), f'{CFG.out_dir}/visualbert_pretrain.pth')
        patience = CFG.patience

    else:
        patience -= 1
        if patience == 0:
            break

