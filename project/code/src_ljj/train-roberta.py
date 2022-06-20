#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import pickle
import random
import numpy as np
import pandas as pd
import time
import jieba

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

from callback.lr_scheduler import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM, PGD
from callback.ema import EMA
from tools.common import seed_everything
from tools.common import init_logger, logger

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import WEIGHTS_NAME, BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer

from models.nezha.modeling_nezha import NeZhaForSequenceClassification, NeZhaModel
from models.nezha.configuration_nezha import NeZhaConfig


# In[2]:


# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss
    for well-classified examples (p>0.5) putting more
    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index,
    should be specific when alpha is float
    :param size_average: (bool, optional) By default,
    the losses are averaged over each loss element in the batch.
    """
    def __init__(self, num_class, alpha=None, gamma=2,
                smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# In[3]:


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert-base-chinese': (BertConfig, AutoModel, BertTokenizer),
    'roberta-base':(BertConfig, AutoModel, BertTokenizer),
    'nezha-cn-base': (NeZhaConfig, NeZhaModel, BertTokenizer),
}


# # CFG

# In[4]:


class CFGs:
    def __init__(self):
        super(CFGs, self).__init__()
        
        self.data_dir = 'data/contest_data'
        self.out_dir = 'data/best_model'
        self.tmp_data_path = 'data/tmp_data'
        
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

        self.batch_size = 512 #16,32
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
        self.patience = 3
        
        self.do_fgm = False
        self.do_pgd = False
        self.do_freelb = False
        self.do_ema = True

        self.log_name = 'data/tmp_data'

        self.overwrite_output_dir = True
        
CFG = CFGs()


# In[ ]:





# In[5]:


task_name = 'roberta'
CFG.tmp_data_path = os.path.join(CFG.tmp_data_path, f'{CFG.model_name}' + task_name)
if not os.path.exists(CFG.tmp_data_path):
    os.makedirs(CFG.tmp_data_path)


# In[6]:


time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
CFG.log_name = os.path.join(CFG.tmp_data_path, f'{task_name}_{time_}.log' )

init_logger(log_file=CFG.log_name)


# In[7]:



# In[8]:


# 记录训练参数
def prn_obj(obj):
    logger.info('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
    
prn_obj(CFG)


# In[9]:


seed_everything(CFG.seed)


# In[10]:


config_class, model_class, tokenizer_class = MODEL_CLASSES[CFG.model_name]


# In[11]:


config = config_class.from_pretrained(CFG.tokenizer_path)
tokenizer = tokenizer_class.from_pretrained(CFG.tokenizer_path)
#bert_model = model_class.from_pretrained(CFG.model_path, config=config)


# In[12]:


CFG.tokenizer = tokenizer

del tokenizer
gc.collect()


# # Read In Data

# In[13]:


data_dir = 'data/contest_data/train_fine.txt'
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
        # 纯色灰色款拉链款加绒裤2021年冬季直筒裤男装
        if ret[0] in ['松紧', '拉链', '系带']:
            if '裤' in title and attr == '裤门襟':
                return attr, ret[0]
            elif ('鞋' in title or '靴' in title) and attr == '闭合方式':
                return attr, ret[0]
            return 'N',''
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
    # 系带进行处理
    if not key_attr:
        return '无'     
    return key_attr #['衣长':'中长款']

img_name = []
img_features = []
texts =[]
key_attr = []
labels = []
class_name = ['图文', '版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']

with open(data_dir, 'r') as f:
    for data in tqdm(f):
        data = json.loads(data)
        img_features.append(np.array(data['feature']).astype(np.float32))
        img_name.append(data['img_name'])
        texts.append(data['title'])
        ## 构造标签
        match = extract_all_key_attr(data['title'])
        key_attr.append(match)
        keys = match.keys()
        # 图文标签为1
        sample_encode = [1]
        # 遍历class_name中的其他关键属性
        for name in class_name[1:]:
            encode = [-1]
            if name in keys: #该属性匹配
                encode = [1]
            sample_encode += encode
        # sample_encode为最后的标签
        labels.append(sample_encode)

coarse_path = 'data/contest_data/train_coarse.txt'
with open(coarse_path, 'r') as f:
    for data in tqdm(f):
        data = json.loads(data)
        if data['match']['图文'] == 1:
            img_features.append(np.array(data['feature']).astype(np.float32))
            img_name.append(data['img_name'])
            texts.append(data['title'])
            ## 构造标签
            match = extract_all_key_attr(data['title'])
            key_attr.append(match)
            keys = match.keys()
            # 图文标签为1
            sample_encode = [1]
            # 遍历class_name中的其他关键属性
            for name in class_name[1:]:
                encode = [-1]
                if name in keys: #该属性匹配
                    encode = [1]
                sample_encode += encode
            # sample_encode为最后的标签
            labels.append(sample_encode)
        
df = pd.DataFrame(img_name)
df['feature'] = img_features
df['text'] = texts
df['key_attr'] = key_attr
df['labels'] = labels
df.columns = ['img_name', 'feature', 'text', 'key_attr', 'labels']


# In[14]:


df.shape


# In[15]:


df = df[df.img_name != 'train139054']



# In[17]:


# df = pd.read_feather(CFG.data_dir + 'multi_label/fine.feather')
# display(df.head())
# df.shape


# In[18]:


# df.to_csv('./data/df_sample_0.1.csv', index=False)


# # CV Split

# In[19]:


from sklearn.model_selection import KFold

# kfold = GroupKFold(CFG.folds)
# for fold, (trn_id, val_id) in enumerate(kfold.split(df, groups=df['img_name'])):
#     df.loc[val_id, 'fold'] = fold


# # Dataset

# In[20]:


class textDataset(Dataset):
    def __init__(self, data, index=None):
        super().__init__()
        self.data = data
        #self.set_type = set_type
        self.class_name = ['图文', '版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
        self.synonym_dict = {
            # 领型
            '高领':['半高领', '立领'], '半高领':['高领', '立领'], '立领':['半高领', '高领'],
            '连帽':['可脱卸帽'], '可脱卸帽':['连帽'],
            '翻领':['衬衫领', 'POLO领', '方领', '娃娃领', '荷叶领'],'衬衫领':['翻领', 'POLO领', '方领', '娃娃领', '荷叶领'],
                    'POLO领':['翻领', '衬衫领', '方领', '娃娃领', '荷叶领'],'方领':['翻领', '衬衫领', 'POLO领', '娃娃领', '荷叶领'],
                    '娃娃领':['翻领', '衬衫领', 'POLO领', '方领', '荷叶领'], '荷叶领':['翻领', '衬衫领', 'POLO领', '方领', '娃娃领'],
            # 袖长
            '短袖':['五分袖'], '五分袖':['短袖'],
            '九分袖':['长袖'], '长袖':['九分袖'], 
            # 衣长
            '超短款':['短款', '常规款'], '短款':['超短款', '常规款'], '常规款':['超短款', '短款'],
            '长款':['超长款'],'超长款':['长款'],
            # 版型
            '修身型':['标准型'], '标准型':['修身型'],
            # 裙长
            '短裙': ['超短裙'], '超短裙': ['短裙'],
            '中裙':['长裙'],'长裙':['中裙'],
            # 裤型
            'O型裤':['锥形裤', '哈伦裤', '灯笼裤'], '锥形裤':['O型裤', '哈伦裤', '灯笼裤'],
            '哈伦裤':['锥形裤', 'O型裤', '灯笼裤'], '灯笼裤':['锥形裤', '哈伦裤', 'O型裤'],
            '铅笔裤':['直筒裤', '小脚裤'], '直筒裤':['铅笔裤', '小脚裤'],  '小脚裤':['直筒裤', '铅笔裤'],
            '喇叭裤':['微喇裤'], '微喇裤':['喇叭裤'],
            # 裤长
            '九分裤':['长裤'], '长裤':['九分裤'],
            # 闭合方式
            '套筒':['套脚', '一脚蹬'], '套脚':['套筒', '一脚蹬'], '一脚蹬':['套筒', '套脚'],
            # 鞋帮高度
            '高帮':['中帮'], '中帮':['高帮']  
        }
        
        self.class_dict = {'图文': ['符合','不符合'], 
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
        
        self.kind_dict = {
            '衣':['针织衫', '外套', '衬衫', '羽绒服', '吊带', '棉服','T恤',
                  '风衣', '仿皮皮衣', '羊毛衫','卫衣', '真皮皮衣', '大衣', 'POLO衫',
                  '毛衣', '连衣裙', '打底衫', '雪纺衫', '羊绒衫', '夹克', '皮草', '马甲',
                  '派克服', '皮衣', '衬衣','背心','棉衣','套装裙'],
            '裤':['牛仔裤', '正装裤', '加绒裤', '休闲裤', '卫裤', '保暖裤', '西装裤', '格子裤子', '运动裤', '垮裤', '西裤', '裙子'],
            '鞋':['休闲鞋', '帆布鞋', '登山鞋', '工装鞋', '运动鞋', '篮球鞋', '板鞋', '皮鞋', '靴子', '雨鞋', '布鞋', '高跟鞋', '童鞋', '雪地靴'],
        }
        self.label_dict = {label:i for i, label in enumerate(class_name)}
        for key, value in self.class_dict.items():
            for key_attr in value:
                jieba.add_word(key_attr)
        
    def __len__(self):
        return self.data.shape[0]
    
    def get_key_attr_neg_single(self, text, key_attr):
        label = [0]*13 # 图文为0
        flag = 0
        class_name_ = self.class_name[1:].copy()
        random.shuffle(class_name_)
        for name in class_name_: #self.class_name： ['图文', '版型', '裤型', ]
            encode = -1 # 如果该关键属性在文本的关键属性中没有的话，该关键属性标签为-1
            keys = list(key_attr.keys())
            random.shuffle(keys)
            for key in keys: # 匹配该text里面的所有关键属性 # key_attr['图文'：xx, '版型'：xx, '裤型':xx, ]
                if key == name:  # 如果和文本的关键属性匹配上，就进行负样本替换
                    encode = 1  # 匹配上了，由于下面可能存在不替代该关键词，所有设置标签为1
                    if flag == 0:
                        val = key_attr[key]  # 匹配上的关键属性的具体取值
                        if val in text: # 如果关键属性中的值在文本中，说明
                            #属性值在texts中，用另外的值替换掉text中文本,
                            tmp = self.class_dict[key]
                            tmp_1 = []
                            for j in tmp:
                                if j != val:
                                    tmp_1.append(j)
                            # 删除同义词替换为负样本
                            if val in self.synonym_dict:
                                for synonym in self.synonym_dict[val]:
                                    tmp_1.remove(synonym)
                            sample = random.choice(tmp_1)
                            #print(val,sample)
                            text = text.replace(val, sample)
                            encode = 0
                            flag = 1
            label[self.label_dict[name]] = encode
        return text, label
    
    def delete_key_attr_neg_single(self, text, key_attr):
        label = [1] + [0]*12 # 图文为0
        flag = 0
        class_name_ = self.class_name[1:].copy()
        random.shuffle(class_name_)
        for name in class_name_: #self.class_name： ['图文', '版型', '裤型', ]
            encode = -1 # 如果该关键属性在文本的关键属性中没有的话，该关键属性标签为-1
            keys = list(key_attr.keys())
            random.shuffle(keys)
            for key in keys: # 匹配该text里面的所有关键属性 # key_attr['图文'：xx, '版型'：xx, '裤型':xx, ]
                if key == name:  # 如果和文本的关键属性匹配上，就进行负样本替换
                    encode = 1  # 匹配上了，由于下面可能存在不替代该关键词，所有设置标签为1
                    if flag == 0:
                        val = key_attr[key]  # 匹配上的关键属性的具体取值
                        if val in text: # 如果关键属性中的值在文本中，说明
                            #属性值在texts中，用另外的值替换掉text中文本,
                            text = ''.join(text.split(val))
                            encode = 0
                            flag = 1
            label[self.label_dict[name]] = encode
        return text, label
    
    def get_key_attr_neg_multi(self, text, key_attr):
        label = [0]*13 # 图文为0
        flag = 0
        class_name_ = self.class_name[1:].copy()
        random.shuffle(class_name_)
        for name in class_name_: #self.class_name： ['图文', '版型', '裤型', ]
            encode = -1 # 如果该关键属性在文本的关键属性中没有的话，该关键属性标签为-1
            keys = list(key_attr.keys())
            random.shuffle(keys)
            for key in keys: # 匹配该text里面的所有关键属性 # key_attr['图文'：xx, '版型'：xx, '裤型':xx, ]
                if key == name:  # 如果和文本的关键属性匹配上，就进行负样本替换
                    val = key_attr[key]  # 匹配上的关键属性的具体取值
                    encode = 1  # 匹配上了，由于下面可能存在不替代该关键词，所有设置标签为1
                    if flag == 0:
                        if val in text: # 如果关键属性中的值在文本中，说明
                            #属性值在texts中，用另外的值替换掉text中文本,
                            tmp = self.class_dict[key]
                            tmp_1 = []
                            for j in tmp:
                                if j != val:
                                    tmp_1.append(j)
                            # 删除同义词替换为负样本
                            if val in self.synonym_dict:
                                for synonym in self.synonym_dict[val]:
                                    tmp_1.remove(synonym)
                            sample = random.choice(tmp_1)
                            #print(val,sample)
                            text = text.replace(val, sample)
                            encode = 0
                            multi_replace_pro = random.random()
                            if multi_replace_pro < 0.3: # 0.3概率不替代了，0.7继续替代
                                flag = 1
                                
            label[self.label_dict[name]] = encode
        return text, label
    
    def get_kinds_neg(self, text, label):
        kind_dict_keys = list(self.kind_dict.keys())
        random.shuffle(kind_dict_keys)
        for key in kind_dict_keys:
            for kind in self.kind_dict[key]:
                if kind.lower() in text.lower():
                    neg = kind.lower()
                    while neg == kind.lower():
                        neg = random.choice(self.kind_dict[key])
                    if random.choice([0, 1]):
                        text = text.replace(kind, neg.lower())
                        text = text.replace(kind.lower(), neg.lower())

                    else:
                        text = text.replace(kind, neg)
                        text = text.replace(kind.lower(), neg)
                    label[0] = 0
        return text, label
    
    def get_synonym_pos(self, text, key_attr):
        # 要用的话要改
        label = [1] 
        for name in self.class_name[1:]: #self.class_name： ['图文', '版型', '裤型', ]
            encode = [-1] # 如果该关键属性在文本的关键属性中没有的话，该关键属性标签为-1
            for key in key_attr.keys(): # 匹配该text里面的所有关键属性 # key_attr['图文'：xx, '版型'：xx, '裤型':xx, ]
                if key == name:  # 如果和文本的关键属性匹配上，就进行同义词替换
                    val = key_attr[key]  # 匹配上的关键属性的具体取值
                    if val in text and val in self.synonym_dict: # 如果关键属性中的值在文本中且有同义词，则进行同义词替换
                        tmp_1 = self.synonym_dict[val] # 同义词表
                        sample = random.choice(tmp_1)
                        text = text.replace(val, sample)
                    encode = [1]
            label += encode
        return text, label
    
    def __getitem__(self, idx):
        
        label = self.data['labels'][idx].copy()
        text = self.data['text'][idx]
        key_attr = self.data['key_attr'][idx]
        feature = torch.tensor(self.data['feature'][idx]).float()
        
        # 构造负样本
        # 1. kinds替代 2. 关键属性替代
        replace_pro = random.random()
        # 4:1.5:1.5:2.5
            
        if replace_pro <= 0.5:
            if replace_pro < 0.2:
                text, label = self.get_key_attr_neg_single(text, key_attr)
            elif replace_pro < 0.35:
                text, label = self.get_key_attr_neg_multi(text, key_attr)
            else:
                text, label = self.get_kinds_neg(text, label)
#         else:
#             if random.random() <= 0.1:
#                 text, label = self.delete_key_attr_neg_single(text, key_attr)
        
        if random.random() <= 0.3:
            text_ = jieba.lcut(text,cut_all=False,HMM=False)
            random.shuffle(text_)
            text= ''.join(text_)
            
        return text, feature, np.array(label) 


# # Model

# In[21]:


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
    
class ITM_Model(nn.Module):
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


# # Helper Function

# In[22]:


def get_optimizer(model, CFG):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {'params': [p for n, p in model.transformer.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': CFG.transformer_lr, 'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in model.transformer.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': CFG.transformer_lr, 'weight_decay': 0.0},
        
            {'params': [p for n, p in model.named_parameters() if "transformer" not in n and not any(nd in n for nd in no_decay)],
             'lr': CFG.clf_lr, 'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in model.named_parameters() if "transformer" not in n and any(nd in n for nd in no_decay)],
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

# In[23]:


def run(df, CFG):
    scores = []
    kf = KFold(n_splits=5)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(df)):
        logger.info('\n')
        logger.info('='*10 + f'fold:{fold}' +'='*10)
        logger.info('\n')
        
        train = df.iloc[trn_idx].reset_index(drop=True)
        valid = df.iloc[val_idx].reset_index(drop=True)
        
        logger.info(f'train on {len(train)} samples, valid on {len(valid)} samples')
        
        logger.info(f'define train_dataset and valid_dataset')
        train_dataset, valid_dataset = textDataset(train), textDataset(valid)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 2, shuffle=False,  pin_memory=True)
        
        logger.info(f'define model')
        model = ITM_Model(CFG).to(device)

        logger.info(f'define optimizer and scheduler')
        optimizer = get_optimizer(model, CFG)
        scheduler = get_scheduler(CFG, optimizer, int(len(train) / CFG.batch_size * CFG.epochs))

#         criterion = nn.BCEWithLogitsLoss()
        criterion = torch.nn.MultiLabelSoftMarginLoss()
        #criterion = FocalLoss(num_class=2)
        if CFG.do_ema:
            ema = EMA(model, 0.999)
            ema.register()
            
        logger.info(f'start training')
        
        best_acc = 0
        best_loss =9999
        
        for epoch in range(CFG.epochs):
            logger.info(f'Epoch:{epoch}')
            
            text_image_neg_size = 1e-10
            text_image_pos_size = 1e-10
            text_image_neg_acc = 0
            text_image_pos_acc = 0

            key_attr_neg_size = 1e-10
            key_attr_pos_size = 1e-10
            key_attr_neg_acc = 0
            key_attr_pos_acc = 0
            
            model.train()
            bar = tqdm(train_loader, total=len(train_loader))
            for text, feature, label in bar: # , max_length=CFG.max_len
                text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
                for k, v in text.items():
                    text[k] = v.to(device)
                    
                img = feature.to(device)
                label = label.to(device)

                ones = torch.ones(label.shape).cuda()
                zeros = torch.zeros(label.shape).cuda()
                
                optimizer.zero_grad()
                outputs = model(text, img).squeeze(1)
                loss = criterion(outputs, torch.where(label==1, ones, zeros))
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
                
                if CFG.do_fgm:
                    #model.zero_grad()
                    fgm = FGM(model, epsilon=0.2, emb_name='word_embeddings.')
                    fgm.attack()
                    logits_fgm = model(text, img).squeeze(1)
                    loss_adv = criterion(logits_fgm, label)
                    loss_adv.backward()
                    fgm.restore()
                if CFG.do_pgd:
                    #model.zero_grad()
                    pgd = PGD(model, emb_name='word_embeddings.', epsilon=1.0,alpha=0.3)
                    K = 3
                    pgd.backup_grad()
                    # 对抗训练
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K-1:
                            model.zero_grad()
                        else:
                            pgd.restore_grad()
                        loss_adv = model(text, img).squeeze(1)
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore() # 恢复embedding参数
                    
                    
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                if CFG.do_ema:
                    ema.update()
                    
                index = (nn.Sigmoid()(outputs) >= 0.5) == label
                text_image_neg_size += (label[:, 0] == 0).sum().item()
                text_image_pos_size += (label[:, 0] == 1).sum().item()
                text_image_neg_acc += (label[:, 0][index[:, 0]] == 0).sum().item()
                text_image_pos_acc += (label[:, 0][index[:, 0]] == 1).sum().item()

                key_attr_neg_size += (label[:, 1:] == 0).sum().item()
                key_attr_pos_size += (label[:, 1:] == 1).sum().item()
                key_attr_neg_acc += (label[:, 1:][index[:, 1:]] == 0).sum().item()
                key_attr_pos_acc += (label[:, 1:][index[:, 1:]] == 1).sum().item()

                text_image_epoch_neg_acc = text_image_neg_acc / text_image_neg_size
                text_image_epoch_pos_acc = text_image_pos_acc / text_image_pos_size
                text_image_epoch_acc = (text_image_neg_acc + text_image_pos_acc) / (text_image_pos_size + text_image_neg_size)
                key_attr_epoch_neg_acc = key_attr_neg_acc / key_attr_neg_size
                key_attr_epoch_pos_acc = key_attr_pos_acc / key_attr_pos_size
                key_attr_epoch_acc = (key_attr_neg_acc + key_attr_pos_acc) / (key_attr_pos_size + key_attr_neg_size)
                epoch_acc = 0.5 * text_image_epoch_acc + 0.5 * key_attr_epoch_acc

                bar.set_postfix(Epoch=epoch,
                                LR=optimizer.param_groups[0]['lr'], 
                                Train_acc=epoch_acc, 
                                text_image_epoch_neg_acc=text_image_epoch_neg_acc,
                                text_image_epoch_pos_acc=text_image_epoch_pos_acc,
                                text_image_epoch_acc = text_image_epoch_acc,
                                key_attr_epoch_neg_acc=key_attr_epoch_neg_acc,
                                key_attr_epoch_pos_acc=key_attr_epoch_pos_acc,
                                key_attr_epoch_acc = key_attr_epoch_acc,
                              )
            
            if CFG.do_ema:
                ema.apply_shadow()
                
            logger.info("***** Train results %s *****")
#             info = f'Train: Epoch_{epoch}_loss: {epoch_loss:.4f}'
#             logger.info(info)
            
            dataset_size = 0
            
            running_loss = 0
            running_acc = 0
            neg_acc = 0
            pos_acc = 0
            neg_size = 1e-10
            pos_size = 1e-10
            bar = tqdm(valid_loader, total=len(valid_loader))
            
            model.eval()
            with torch.no_grad():
                for text, feature, label in bar:# max_len
                    text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
                    for k, v in text.items():
                        text[k] = v.to(device)
                    img = feature.to(device)
                    label = label.to(device)
                    
                    ones = torch.ones(label.shape).cuda()
                    zeros = torch.zeros(label.shape).cuda()
                    
                    outputs = model(text, img).squeeze(1)
                    loss = criterion(outputs, torch.where(label==1, ones, zeros))

                    index = (nn.Sigmoid()(outputs) >= 0.5) == label
                    text_image_neg_size += (label[:, 0] == 0).sum().item()
                    text_image_pos_size += (label[:, 0] == 1).sum().item()
                    text_image_neg_acc += (label[:, 0][index[:, 0]] == 0).sum().item()
                    text_image_pos_acc += (label[:, 0][index[:, 0]] == 1).sum().item()

                    key_attr_neg_size += (label[:, 1:] == 0).sum().item()
                    key_attr_pos_size += (label[:, 1:] == 1).sum().item()
                    key_attr_neg_acc += (label[:, 1:][index[:, 1:]] == 0).sum().item()
                    key_attr_pos_acc += (label[:, 1:][index[:, 1:]] == 1).sum().item()

                    text_image_epoch_neg_acc = text_image_neg_acc / text_image_neg_size
                    text_image_epoch_pos_acc = text_image_pos_acc / text_image_pos_size
                    text_image_epoch_acc = (text_image_neg_acc + text_image_pos_acc) / (text_image_pos_size + text_image_neg_size)
                    key_attr_epoch_neg_acc = key_attr_neg_acc / key_attr_neg_size
                    key_attr_epoch_pos_acc = key_attr_pos_acc / key_attr_pos_size
                    key_attr_epoch_acc = (key_attr_neg_acc + key_attr_pos_acc) / (key_attr_pos_size + key_attr_neg_size)
                    epoch_acc = 0.5 * text_image_epoch_acc + 0.5 * key_attr_epoch_acc

                    bar.set_postfix(Epoch=epoch,
                                LR=optimizer.param_groups[0]['lr'], 
                                Train_acc=epoch_acc, 
                                text_image_epoch_neg_acc=text_image_epoch_neg_acc,
                                text_image_epoch_pos_acc=text_image_epoch_pos_acc,
                                text_image_epoch_acc = text_image_epoch_acc,
                                key_attr_epoch_neg_acc=key_attr_epoch_neg_acc,
                                key_attr_epoch_pos_acc=key_attr_epoch_pos_acc,
                                key_attr_epoch_acc = key_attr_epoch_acc,
                              )
            

                
            logger.info("***** Eval results %s *****")
#             info = f'Eval: Epoch_{epoch}_eval_loss: {epoch_loss:.4f}'
#             logger.info(info)
#             logger.info(f'\n')
            
            if epoch_acc > best_acc:
                logger.info(f'Weighted_acc improved: best_acc improved from {best_acc} -----> {epoch_acc}')
                best_acc = epoch_acc
                logger.info(f'\n')
                torch.save(model.state_dict(), f'{CFG.out_dir}/model_roberta_fold{fold}.pth')
                patience = CFG.patience

            else:
                patience -= 1
                if patience == 0:
                    break

            if CFG.do_ema:
                ema.restore()
        scores.append(best_acc)
#         del train, valid,train_dataset, valid_dataset,train_loader,valid_loader, model
#         gc.collect()
#         torch.cuda.empty_cache()
    logger.info(f'avg acc:{np.mean(scores)}')
    logger.info(scores)


# In[24]:


device = CFG.device
run(df, CFG)


# In[ ]:





# In[ ]:





# In[ ]:




