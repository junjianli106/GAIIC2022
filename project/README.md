# 代码说明

## 环境配置

在requirements中，python=3.7, torch=1.8.1, torchvision=0.9.1
在init.sh中，通过pip install -r ./requirements.txt安装依赖包

## 数据

没用使用额外的数据集

## 预训练模型

1. 使用mlm的方式对huggingface transformer的uclanlp/visualbert-vqa-coco-pre模型进行了预训练
2. 使用mlm的方式对huggingface transformer的chinese-roberta-wwm-ext进行预训练
3. 使用mlm的方式对nezha-base-chinese进行预训练，可通过https://github.com/lonePatient/NeZha_Chinese_PyTorch获得。

此外，上述预训练方式都在本赛题的数据上进行了进一步的预训练。

## 算法（必选）

### 整体思路介绍
整体的思路是使用bert模型作为文本特征提取网络，然后将提取之后的特征和图片feature进行concat，concat之后的特征输入到MLP中进行分类。

### 数据扩增
本赛题的关键是如何构造负样本，我们对原始数据中的coarse和fine数据进行合并，并且去除了coarse中的图文不匹配的样本，然后得到了全部为图文匹配的样本。我们在训练过程中，每一个样本都有一定的概率被变为负样本。具体负样本构造可以分为替代单个关键属性，替代多个关键属性和替代隐藏属性

1. 对于单个关键属性的替代：有0.2的概率对该正样本替换单个关键属性。
2. 对于多个关键属性的替代:有0.15的概率对多个关键属性进行替换，此外，在替换关键属性的时候，有0.3的概率跳出。不进行下一次关键属性的替代。
3. 对于隐藏属性的替代:以0.15的概率对隐藏属性进行替代。
4. 对于剩余的0.5的正样本，有0.3的概率使用jieba分词之后进行随机shuffle。

### 模型集成
对roberta，nezha和visualbert进行融合。权重是1：1：1

## 损失函数
对图文是否匹配任务，我们使用二分类对图文是否匹配进行分类，对于其他关键属性，我们也是使用2分类来分类图片和该关键属性是否匹配。所以我们使用的是MultiLabelSoftMarginLoss

## 训练流程

1. 对原始数据进行预处理,生成训练和测试能够使用的数据（其实就是提取所有样本的关键属性）。
python ./code/src_lcc/utils/process_train_data.py
python ./code/src_lcc/utils/process_test_data.py

2. 预训练（可不选）
python ./code/src_lcc/pretrain-visualbert.py

3. 对visualbert训练
python ./code/src_lcc/train-multitask-visualbert.py

4. 训练roberta和nezha
python ./code/src_ljj/train-roberta.py
python ./code/src_ljj/train-nezha.py

## 测试流程
1. visualbert推理
python ./code/src_lcc/utils/process_test_data.py 'B'
python ./code/src_lcc/inference-visualbert-multitask.py

2. roberta，nezha推理和最终融合
python ./code/src_ljj/inference.py ./data/contest_data/preliminary_testB.txt



