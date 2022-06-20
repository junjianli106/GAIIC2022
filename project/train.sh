# 生成训练数据
python ./code/src_lcc/utils/process_train_data.py
python ./code/src_lcc/utils/process_test_data.py

# 预训练
#python ./code/src_lcc/pretrain-visualbert.py
# 对visualbert训练
python ./code/src_lcc/train-multitask-visualbert.py

# 训练roberta和nezha
python ./code/src_ljj/train-roberta.py
python ./code/src_ljj/train-nezha.py