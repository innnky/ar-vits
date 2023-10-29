import numpy
import pandas

semantic_path = 'dump/semantic.tsv'
phoneme_path = 'dump/phoneme.npy'
train_semantic_path = 'dump/semantic_train.tsv'
train_phoneme_path = 'dump/phoneme_train.npy'
dev_semantic_path = 'dump/semantic_dev.tsv'
dev_phoneme_path = 'dump/phoneme_dev.npy'

# 读取dump/semantic.tsv
semantic_df = pandas.read_csv(semantic_path, sep='\t')
# pd.DataFrame(columns=["item_name", "semantic_audio"])
# # 读取dump/phoneme.npy
phoneme_dict = numpy.load(phoneme_path, allow_pickle=True).item()

dev_num = 20
# 随机从semantic_df中选取dev_num个
dev_df = semantic_df.sample(n=dev_num)
# 剩下的是train
train_df = semantic_df.drop(dev_df.index)
# 保存
dev_df.to_csv(dev_semantic_path, sep='\t', index=False)
train_df.to_csv(train_semantic_path, sep='\t', index=False)

# 将dev_df中的item_name取出来 作为dev_phoneme_dict的key
dev_item_names = dev_df['item_name'].tolist()
dev_phoneme_dict = {k: phoneme_dict[k] for k in dev_item_names if k in phoneme_dict}
train_phoneme_dict = {k: phoneme_dict[k] for k in phoneme_dict.keys() if k not in dev_item_names}

numpy.save(dev_phoneme_path, dev_phoneme_dict)
numpy.save(train_phoneme_path, train_phoneme_dict)



