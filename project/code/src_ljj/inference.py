#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import sys
import pickle
import random
import warnings
import pickle
import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from callback.lr_scheduler import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM

from tools.common import seed_everything
from tools.common import init_logger, logger

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer

from models.nezha.modeling_nezha import NeZhaForSequenceClassification, NeZhaModel
from models.nezha.configuration_nezha import NeZhaConfig

# from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei']

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


warnings.filterwarnings('ignore')


# In[2]:


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, AutoModel, BertTokenizer),
    'roberta-base':(BertConfig, AutoModel, BertTokenizer),
    'nezha-cn-base': (NeZhaConfig, NeZhaModel, BertTokenizer),
}


# In[3]:


class CFGs:
    def __init__(self):
        super(CFGs, self).__init__()
        
        self.data_dir = 'data/'
        self.out_dir = 'data/submission'
        self.best_model_path = 'data/best_model'
        self.tmp_data = 'data/tmp_data'
        
        self.epochs=100
        self.folds = 5

        self.task = 'whole' # whole detail
        #self.train_file = f'{self.task}.pkl'
        self.train_file = f'{self.task}.pkl'
        self.model_name = 'roberta-base'
        self.tokenizer_path = 'data/pretrain_model/chinese-roberta-wwm-ext'
        self.model_path = 'data/pretrain_model/chinese-roberta-wwm-ext'

        self.scheduler='cosine'
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 64 #16,32
        self.dropout = 0.2
        #self.max_len = 40

        self.text_dim = 768
        self.img_dim = 2048

        self.transformer_lr = 2e-5
        self.clf_lr = 1e-4

        self.weight_decay = 0.01
        self.eps=1e-6
        self.betas=(0.9, 0.999)
        self.num_warmup_steps=0

        self.max_norm = 1000
        self.num_cycles=0.5
        self.patience = 5
        
        self.do_fgm = False
        self.do_pgd = False
        self.do_freelb = False
        self.do_ema = True

        self.log_name = './output'

        self.overwrite_output_dir = True
        
CFG = CFGs()


# In[4]:


config_class, model_class, tokenizer_class = MODEL_CLASSES[CFG.model_name]
    
config = config_class.from_pretrained(CFG.model_path)
tokenizer = tokenizer_class.from_pretrained(CFG.tokenizer_path)
bert_model = model_class.from_pretrained(CFG.model_path, config=config)
CFG.tokenizer = tokenizer

del tokenizer
gc.collect()


# In[5]:


class MyDataset(Dataset):
    def __init__(self, df):
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
            return self.text[index],  torch.tensor(self.feature[index]), torch.tensor(self.label[index]).long()
        else:
            return self.text[index],  torch.tensor(self.feature[index])


# In[6]:


class FuseLayer(nn.Module):
    def __init__(self, text_dim, img_dim, dropout):
        super().__init__()
        self.bn = nn.BatchNorm1d(768*2)
        self.fc1 = nn.Sequential(
            nn.Linear(img_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

        )
    
    def forward(self, text, img):
        img = self.fc1(img)
        
        concat = torch.cat((img, text),dim=1)
        concat = self.bn(concat)
        fuse = self.fc2(concat)
        return fuse
    
class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        dropout = CFG.dropout
        self.transformer = model_class.from_pretrained(CFG.model_path, config=config)
        self.dropout = nn.Dropout(dropout)
        self.fuse = FuseLayer(CFG.text_dim, CFG.img_dim, dropout)
        self.clf = nn.Linear(CFG.text_dim, 2)
        self.clf1 = nn.Sequential(
                    nn.Linear(CFG.text_dim, 256),
                    nn.Linear(256, 64),
                    nn.Linear(64, 13))
        
    def forward(self, text, img):
        text = self.transformer(**text)[1]
        text = self.dropout(text)
        fuse = self.fuse(text, img)
        out = self.clf1(fuse)
        return out


# In[7]:


device = CFG.device
# task_name = '_pretrained_6000_2_1.5_1.5_5_shuffle_0.3_fold5_修正数据集'
# CFG.out_dir = os.path.join(CFG.out_dir, f'roberta-base' + task_name)


# In[8]:
import sys

data_dir = sys.argv[1]#'../../data/contest_data/preliminary_testB.txt'

import json
import itertools
import re
def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r') as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

attr_dict_file = "data/contest_data/attr_to_attrvals.json"
attr_dict = load_attr_dict(attr_dict_file)

def extract_key_attr(title, attr, attr_dict):
    # 在title中匹配属性值
    if attr == '图文':
        return '图文', '符合'
    attr_dict1 = attr_dict
    attrvals = "|".join(attr_dict1[attr])
    ret = re.findall(attrvals, title)
    if ret:
        return attr, ret[0]
    else:
        return 'N',''


def extract_all_key_attr(text):
    key_attr = {}
    for attr in class_name:
        #print(text, attr)
        ret_attr, class_label = extract_key_attr(text, attr, attr_dict)
        if ret_attr != 'N':
            key_attr[ret_attr] = class_label
    if not key_attr:
        return '无'
    return key_attr #['衣长':'中长款']

img_name = []
img_features = []
texts =[]
querys = []
class_name = ['图文', '版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']

with open(data_dir, 'r') as f:
    for data in tqdm(f):
        data = json.loads(data)
        img_features.append(np.array(data['feature']).astype(np.float32))
        img_name.append(data['img_name'])
        texts.append(data['title'])
        querys.append(data['query'])

df = pd.DataFrame(img_name)
df['feature'] = img_features
df['text'] = texts
df['querys'] = querys
df.columns = ['img_name', 'feature', 'text', 'querys']


# In[9]:


df.head()


# In[10]:


@torch.no_grad()
def evaluate(df, CFG, fold):
    pred = np.empty((0,13))
    dataset = MyDataset(df)
    loader = DataLoader(dataset, batch_size=CFG.batch_size, pin_memory=True)
    model = Model(CFG).to(device)
    model.eval()
    model.load_state_dict(torch.load(f'{CFG.best_model_path}/model_roberta_fold{fold}.pth'))
    for text, feature in tqdm(loader):
        text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        for k, v in text.items():
            text[k] = v.cuda()
        img = feature.cuda()
        outputs = model(text, img)
        outputs = torch.sigmoid(outputs)
        #print(outputs.shape, outputs)
        pred= np.concatenate((pred, outputs.cpu().numpy()))
        #print(pred.shape)
        #pred = np.stack([pred, outputs.cpu().numpy()])
    return pred


# In[11]:


pred = []
CFG.folds = 5

for fold in range(CFG.folds):
    pred.append(evaluate(df, CFG, fold))



# pred = []
# CFG.folds = 1

# for fold in [2]:
#     pred.append(evaluate(df, CFG, fold))

# pred = np.mean(pred, axis=0)


# In[12]:


pred = np.mean(pred, axis=0)


# In[13]:


pred


# In[14]:


CFG.model_name = 'nezha-cn-base'
CFG.tokenizer_path = 'data/pretrain_model/nezha-cn-base'
CFG.model_path = 'data/pretrain_model/nezha-cn-base'


# In[15]:


config_class, model_class, tokenizer_class = MODEL_CLASSES[CFG.model_name]
    
config = config_class.from_pretrained(CFG.model_path)
tokenizer = tokenizer_class.from_pretrained(CFG.tokenizer_path)
bert_model = model_class.from_pretrained(CFG.model_path, config=config)
CFG.tokenizer = tokenizer

del tokenizer
gc.collect()


# In[16]:


device = CFG.device
# task_name = '_pretrained_6000_2_1.5_1.5_5_shuffle_0.3_fold5_修正数据集'
# CFG.out_dir = os.path.join('./output', f'nezha-cn-base' + task_name)


# In[17]:


@torch.no_grad()
def evaluate(df, CFG, fold):
    pred = np.empty((0,13))
    dataset = MyDataset(df)
    loader = DataLoader(dataset, batch_size=CFG.batch_size, pin_memory=True)
    model = Model(CFG).to(device)
    model.eval()
    model.load_state_dict(torch.load(f'{CFG.best_model_path}/model_nezha_fold{fold}.pth', map_location='cpu'), strict=False)
    for text, feature in tqdm(loader):
        text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
        for k, v in text.items():
            text[k] = v.cuda()
        img = feature.cuda()
        outputs = model(text, img)
        outputs = torch.sigmoid(outputs)
        #print(outputs.shape, outputs)
        pred= np.concatenate((pred, outputs.cpu().numpy()))
        #print(pred.shape)
        #pred = np.stack([pred, outputs.cpu().numpy()])
    return pred


# In[18]:


pred_nezha = []
CFG.folds = 5

for fold in range(CFG.folds):
    pred_nezha.append(evaluate(df, CFG, fold))


# In[19]:


pred_nezha = np.mean(pred_nezha, axis=0)


# In[20]:

pred_cc = pd.read_csv(os.path.join(CFG.tmp_data, 'pred.csv'))
pred_cc = pred_cc[['图文', '版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']]
pred_cc = pred_cc.values


# In[21]:


pred_last = np.mean([pred, pred_nezha, pred_cc],  axis=0)


df['pred'] = list(pred_last)


# In[24]:


df.head()


# In[25]:


class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
class_dict={'图文': ['符合','不符合'], 
            '版型': ['修身型', '宽松型', '标准型'], 
            '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤', '紧身裤'],
            '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'], 
            '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'], 
            '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领', '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'], 
            '裤门襟': ['系带', '松紧', '拉链'], 
            '鞋帮高度': ['低帮', '高帮', '中帮'], 
            '穿着方式': ['套头', '开衫'], 
            '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'], 
            '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'], 
            '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'], 
            '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
            }


# In[26]:


df.head()


# In[27]:


df['pred_0'] = df['pred'].apply(lambda x:x[0])


# In[28]:


def function(x):
    query = x['querys']
    pre = x['pred']
    tmp={}
    for que in query:
#         if que != '图文':
#             tmp[que]=2
#             continue
        inx=class_name.index(que)
        if pre[inx] > Threshold:
            #print(pre[inx])
            tmp[que]=1
        else:
            tmp[que]=0
    return tmp


# In[29]:


# Threshold = sorted(list(df['pred_0'].values))[2326]
# print(Threshold)
Threshold = 0.5
df['match'] = df.apply(function, axis=1)


# In[30]:


def itm_same_as_tuwen(x):
    ret = {}
    for key, value in x['match'].items():
        if key == '图文':
            ret[key] = value
        elif ret['图文'] and ret['图文'] == 1:
            ret[key] = 1
        else:
            ret[key] = value
    return ret


# In[31]:


def itm_same_as_key_attr(x):
    ret = {}
    for key, value in x['match'].items():
        if value == 0:
            ret['图文'] = 0
        ret[key] = value
    return ret


# In[32]:


# # df['itm'] = df['match'].apply(lambda x:x['图文'])
# df['match'] = df.apply(itm_same_as_key_attr, axis=1)


# In[33]:


df.head()


# In[38]:


def count_key0_itm1(x):
    for i in x.keys():
        if i == '图文' and x[i]==1:
            flag = 1
        elif i == '图文' and x[i]==0:
            flag = 0
        else:
            if x[i] == 0 and flag == 1:
                return 1
    return 0


# In[39]:


df['key0_itm1'] = df['match'].apply(lambda x:count_key0_itm1(x)) 


# In[40]:


df['key0_itm1'].value_counts()


# In[43]:


def image_item(x):
    ret = {}
    for i in x.keys():
        if i == '图文' and x[i]==1:
            ret['图文'] = 1
        elif i == '图文' and x[i]==0:
            ret['图文'] = 0
        else:
            if x[i] == 0 and ret['图文'] == 1:
                ret[i] = 1
            else:
                ret[i] = x[i]
    return ret


# In[44]:


df['match1'] = df['match'].apply(lambda x:image_item(x))


# In[47]:


# df[df.key0_itm1 == 1]


# In[48]:


submit=[]
submit_sample={"img_name":"test000255","match":{"图文":0,"领型":1,"袖长":1,"穿着方式":0}}
for i, row in df.iterrows():
    submit_sample['img_name']=row['img_name']
    submit_sample['match']=row['match1']
    #print(submit_sample)
    submit.append(json.dumps(submit_sample, ensure_ascii=False)+'\n')


# In[49]:


with open(os.path.join(CFG.out_dir, 'results.txt'), 'w') as f:
    f.writelines(submit)




