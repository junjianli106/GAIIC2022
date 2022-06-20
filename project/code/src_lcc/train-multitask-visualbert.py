#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gc
import torch
import json
import jieba
# import wandb
import pandas as pd
from itertools import chain


# # CFG

# In[2]:


class CFG:
    data_dir = 'data/tmp_data/'
    ori_dir = 'data/contest_data/'
    out_dir = 'data/model_data'
    pretrain_dir = 'data/pretrain_model/visualbert'
    scheduler='cosine'
    model_name = 'uclanlp/visualbert-vqa-coco-pre'
    train_file = 'multitask.feather'
    
    seed = 42
    folds = 5
    batch_size = 128
    dropout = 0.2
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
    
    epochs= 10000
    patience = 5
    ema = True
    
params_dict = {key:value for key, value in CFG.__dict__.items() if not key.startswith('__')}
params_dict


# # Wandb

# In[3]:


# tags=['visualbert-40-pretrain', 'jieba-shuffle']
# wandb.init(project="gaiic2022", config=params_dict, tags=tags, name='')


# In[4]:


labels = ['图文','领型', '袖长', '衣长', '版型', '裙长', '穿着方式', '类别', '裤型', '裤长', '裤门襟', '闭合方式', '鞋帮高度']


# In[5]:


if not os.path.exists(CFG.out_dir):
    os.makedirs(CFG.out_dir)


# In[6]:


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


# In[7]:


from transformers import AutoTokenizer, AutoConfig


# In[8]:


CFG.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


# # Read In Data

# In[9]:


df = pd.read_feather(CFG.data_dir + CFG.train_file)
# display(df.head())
df.shape


# # CV Split

# In[10]:


from sklearn.model_selection import GroupKFold,KFold


# In[11]:


kfold = GroupKFold(CFG.folds)
for fold, (trn_id, val_id) in enumerate(kfold.split(df, groups=df['img_name'])):
    df.loc[val_id, 'fold'] = fold


# # Dataset

# In[12]:


from torch.utils.data import Dataset, DataLoader


# ## Constant

# In[13]:


def load_attr_dict():
    attr_dict = {}
    for attr, attrval_list in key_attr.items():
        attrval_list = list(map(lambda x: x.split('='), attrval_list))
        attrval_list = list(map(lambda x: x[0], attrval_list))
        attr_dict[attr] = attrval_list
    return attr_dict

with open(CFG.ori_dir + 'attr_to_attrvals.json', 'rb') as f:
    key_attr = json.load(f)
key_attr = load_attr_dict()
key_attr # dict without duplicate


# In[14]:


type_dict = dict()
for key,values in key_attr.items():
    for value in values:
        type_dict[value] = key
type_dict


# In[15]:


kinds = {
    '衣':[
        '针织衫', '外套', '衬衫', '羽绒服', '吊带','棉服','T恤',
         '风衣','仿皮皮衣','羊毛衫','卫衣', '真皮皮衣','大衣','POLO衫',
        '毛衣', '连衣裙', '打底衫','雪纺衫','羊绒衫','夹克', '皮草','马甲',
        '派克服','皮衣','衬衣','背心','棉衣','套装裙'
        ],
    '裤':['牛仔裤','正装裤', '加绒裤','休闲裤', '卫裤', '保暖裤', '西装裤','格子裤子', '运动裤', '垮裤','西裤', '裙子'],
    '鞋':['休闲鞋','帆布鞋','登山鞋', '工装鞋', '运动鞋', '篮球鞋','板鞋','皮鞋', '靴子','雨鞋','布鞋','高跟鞋','童鞋', '雪地靴'],
}


# In[16]:


values = list(chain.from_iterable(list(kinds.values()))) + list(chain.from_iterable(list(key_attr.values())))
for value in values:
    jieba.add_word(value)


# In[17]:


def replace_single_key_attr(x, label):
    pos_index = np.where(label == 1)[0][1:].tolist()
    if pos_index:
        selected_index = random.choice(pos_index)
        selected_attr = labels[selected_index]
        for Type in key_attr[selected_attr]:
            if Type in x:
                neg = Type
                while neg == Type:
                    neg = random.choice(key_attr[selected_attr])
                x = x.replace(Type, neg)
                label[selected_index] = 0
                label[0] = 0
                return x, label
    return x, label
        
def replace_multi_key_attr(x, label):
    pos_index = np.where(label == 1)[0][1:].tolist()
    while pos_index:
        selected_index = random.choice(pos_index)
        selected_attr = labels[selected_index]
        for Type in key_attr[selected_attr]:
            if Type in x:
                neg = Type
                while neg == Type:
                    neg = random.choice(key_attr[selected_attr])
                x = x.replace(Type, neg)
                label[selected_index] = 0
                label[0] = 0
                pos_index.remove(selected_index)
                break
        if random.random() < 0.3:
            break
    return x, label

def replace_hidden_attr(x, label):
    for key in kinds.keys():
        for kind in kinds[key]:
            if kind.lower() in x.lower():
                neg = kind.lower()
                while neg == kind.lower():
                    neg = random.choice(kinds[key])
                if random.choice([0,1]):
                    x = x.replace(kind, neg.lower())
                    x = x.replace(kind.lower(), neg.lower())
                else:
                    x = x.replace(kind, neg)
                    x = x.replace(kind.lower(), neg)
                label[0] = 0
                return x, label
    return x, label


# ## MyDataset

# In[18]:


class MyDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.text = df['text'].values
        self.feature = df['feature'].values
        self.label = None
        if '领型' in df.columns:
            self.label = df[labels].values
        self.neg_pro = 0.75
        self.single_pro = 0.2
        self.multi_pro = 0.55
        
        
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        if self.label is not None:
            text = self.text[index]
            label = self.label[index]
            text, label = self.get_neg(text, label.copy())
            return text, torch.tensor(self.feature[index]).float(), torch.tensor(label).float()
        else:
            return self.text[index], torch.tensor(self.feature[index]).float()
            
    def get_neg(self,x, label):
        probs = random.random()
        if  probs <= self.neg_pro:
            if probs <= self.single_pro:
                return replace_single_key_attr(x, label)
            if probs <= self.multi_pro:
                return replace_multi_key_attr(x, label)
            else:
                return replace_hidden_attr(x, label)
        if random.random() < 0.3:
            x = jieba.lcut(x, HMM=False)
            random.shuffle(x)
            x = ''.join(x)
        return x, label


# # Model

# In[19]:


import torch.nn as nn
from transformers import AutoModel, VisualBertModel


# In[20]:


class Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(CFG.model_name)
        self.transformer.load_state_dict(torch.load(f'{CFG.pretrain_dir}/visualbert_pretrain.pth'))
        self.dropout = nn.Dropout(CFG.dropout)
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


# # Helper Function

# In[21]:


from torch.optim import AdamW, Adam
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


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


# In[23]:


from tqdm.auto import tqdm
from termcolor import colored

from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# # EMA

# In[24]:


class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# # Run

# In[25]:


def run():
    scores = []
    for fold in range(CFG.folds):
        print('='*10 + f'fold:{fold}' +'='*10)
        train = df[df['fold'] != fold].reset_index(drop=True)
        valid = df[df['fold'] == fold].reset_index(drop=True)
        print(f'train on {len(train)} samples, valid on {len(valid)} samples')
        train_dataset, valid_dataset = MyDataset(train), MyDataset(valid)
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 4, pin_memory=True)
        
        model = Model(CFG).cuda()
#         wandb.watch(model)
        
        optimizer = get_optimizer(model, CFG)
        scheduler = get_scheduler(CFG, optimizer, int(len(train) / CFG.batch_size * CFG.epochs))

        criterion = nn.MultiLabelSoftMarginLoss()
        best_acc = 0
        if CFG.ema:
            ema = EMA(model)
            ema.register()
        for epoch in range(CFG.epochs):

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
            for text, feature,label in bar:
                text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
                for k, v in text.items():
                    text[k] = v.cuda()
                img = feature.cuda()
                label = label.cuda()
                
                optimizer.zero_grad()
                
                outputs = model(text,img)
                loss = criterion(outputs, torch.where(label==1, 1, 0))

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), CFG.max_norm)
                optimizer.step()
                scheduler.step()

                if CFG.ema:
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

            if CFG.ema:
                ema.apply_shadow()


            text_image_neg_size = 1e-10
            text_image_pos_size = 1e-10
            text_image_neg_acc = 0
            text_image_pos_acc = 0

            key_attr_neg_size = 1e-10
            key_attr_pos_size = 1e-10
            key_attr_neg_acc = 0
            key_attr_pos_acc = 0
            model.eval()
            with torch.no_grad():
                bar = tqdm(valid_loader, total=len(valid_loader))
                for text, feature,label in bar:
                    text = CFG.tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=True)
                    for k, v in text.items():
                        text[k] = v.cuda()
                    img = feature.cuda()
                    label = label.cuda()

                    outputs = model(text,img)

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
                    del text, feature,label
                    gc.collect()
                    torch.cuda.empty_cache()
#                 wandb.log({ 
#                             f'epoch_fold{fold}':epoch,
#                             f'Valid_acc_fold{fold}':epoch_acc, 
#                             f'Valid_text_image_epoch_neg_acc_fold{fold}':text_image_epoch_neg_acc,
#                             f'Valid_text_image_epoch_pos_acc_fold{fold}':text_image_epoch_pos_acc,
#                             f'Valid_text_image_epoch_acc_fold{fold}':text_image_epoch_acc,
#                             f'Valid_key_attr_epoch_neg_acc_fold{fold}':key_attr_epoch_neg_acc,
#                             f'Valid_key_attr_epoch_pos_acc_fold{fold}':key_attr_epoch_pos_acc,
#                             f'Valid_key_attr_epoch_acc_fold{fold}':key_attr_epoch_acc,})

            if epoch_acc > best_acc:
                print(f'best_acc improved from {best_acc} -----> {epoch_acc}')
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"{CFG.out_dir}/model_multitask_fold{fold}_{CFG.model_name.split('/')[-1]}.pth")
                patience = CFG.patience

            else:
                patience -= 1
                if patience == 0:
                    break

            if CFG.ema:
                ema.restore()

        scores.append(best_acc)
        del train, valid,train_dataset, valid_dataset,train_loader,valid_loader, model
        gc.collect()
        torch.cuda.empty_cache()
    print(f'avg acc:{np.mean(scores)}')
    print()
    print(scores)
#     wandb.log({
#         'scores':scores,
#         'avg_score':np.mean(scores),
#     })



run()


