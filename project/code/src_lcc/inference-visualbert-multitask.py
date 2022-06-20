#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import gc
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')


# In[2]:


class CFG:
    data_dir = 'data/tmp_data/'
    model_dir = 'data/best_model'
    scheduler='cosine'
    task = 'whole' # whole detail
    model_name = 'uclanlp/visualbert-vqa-coco-pre'
    model_path = data_dir + 'MaskedLM'
    test_file = f'test_{task}.feather'
    seed = 42
    folds = 5
    batch_size = 16
    dropout = 0.1
    text_dim = 768
    img_dim = 2048
    transformer_lr = 2e-5
    clf_lr = 1e-4
    weight_decay = 0.01
    eps=1e-6
    betas=(0.9, 0.999)
    num_warmup_steps=0
    epochs=5
    max_norm = 1000
    num_cycles=0.5


# In[3]:


from transformers import AutoTokenizer


# In[4]:


CFG.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


# In[5]:


labels = ['图文','领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度']


# In[6]:


from torch.utils.data import Dataset, DataLoader


# In[7]:


class MyDataset(Dataset):
    def __init__(self, df, use_type=False):
        super().__init__()
        self.text = df['text'].values
        self.feature = df['feature'].values
        self.label = None
        if 'label' in df.columns:
            self.label = df['label'].values
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        if self.label is not None:
            return self.text[index],  torch.tensor(self.feature[index]).float(), torch.tensor(self.label[index]).long()
        else:
            return self.text[index],  torch.tensor(self.feature[index]).float()


# In[8]:


import torch
import torch.nn as nn
from transformers import AutoModel


# In[9]:


class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        dropout = CFG.dropout
        self.transformer = AutoModel.from_pretrained(CFG.model_name)
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.Sequential(
                nn.Linear(CFG.text_dim, 512),
                nn.Linear(512, 256),
                nn.Linear(256, len(labels))
        )
            
    def forward(self, text, img):
        text.update({'visual_embeds': img.unsqueeze(1)})
        fuse = self.transformer(**text)[1]
        fuse = self.dropout(fuse)
        out = self.clf(fuse)
        return out


# In[10]:


df = pd.read_feather(CFG.data_dir + CFG.test_file)
df.head()


# In[ ]:


detail = pd.read_feather(CFG.data_dir + 'test_detail.feather', columns=['img_name', 'type'])
detail.head()


# In[ ]:


from tqdm.auto import tqdm


# In[ ]:


@torch.no_grad()
def evaluate(df, CFG, fold):
    pred = np.empty((0,13))
    dataset = MyDataset(df)
    loader = DataLoader(dataset, batch_size=CFG.batch_size * 2, pin_memory=True)
    model = Model(CFG).cuda()
    model.eval()
    model.load_state_dict(torch.load(f"{CFG.model_dir}/model_multitask_fold{fold}_{CFG.model_name.split('/')[-1]}.pth"))
    for text, feature in tqdm(loader):
        text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        for k, v in text.items():
            text[k] = v.cuda()
        img = feature.cuda()
        outputs = nn.Sigmoid()(model(text, img))
        pred= np.concatenate((pred, outputs.cpu().numpy()))
    return pred


# In[ ]:


labels = ['图文','领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度']


# In[ ]:


pred = []
for fold in range(CFG.folds):
    pred.append(evaluate(df, CFG, fold))

merge = pd.DataFrame(np.mean(pred, axis=0),columns=labels)
merge = pd.concat((df, merge), axis=1)
merge = merge[['img_name'] + labels]
merge.to_csv(f'{CFG.data_dir}/pred.csv', index=False)

pred = pd.DataFrame(np.where(np.mean(pred, axis=0)>=0.5, 1, 0),columns=labels)
pred = pd.concat((df, pred), axis=1)
pred.index = pred['img_name']

text_img_pred = pred['图文']
# display(text_img_pred.head())
# display(text_img_pred.value_counts())
text_img_pred.name = 'pred'
text_img_pred.to_csv(f'{CFG.data_dir}/whole_pred.csv')

key_attr_pred = pd.DataFrame([])
labels.remove('图文')
for label in labels:
    tmp = pred[label]
    tmp.name = 'pred'
    tmp = pd.DataFrame(tmp)
    tmp['type'] = label
    key_attr_pred = pd.concat((key_attr_pred, tmp))
key_attr_pred = key_attr_pred[['pred', 'type']]
detail = detail.merge(key_attr_pred, how='left', on=['img_name', 'type'])
# display(detail)
# display(detail.pred.value_counts())
detail.to_csv(f'{CFG.data_dir}/detail_pred.csv')


# In[ ]:


# for label in labels:
#     display(detail[detail.type==label].pred.value_counts())

