from glob import glob
from random import shuffle

data_root = '/home/fish/wenetspeech/dataset_vq'
filenames = glob(f"{data_root}/**/*.flac", recursive=True)  # [:10]

shuffle(filenames)
val_num = 8
train = filenames[:-val_num]
val = filenames[-val_num:]
train.sort()
val.sort()

with open('filelists/train.list', 'w') as f:
    f.write('\n'.join(train))
with open('filelists/val.list', 'w') as f:
    f.write('\n'.join(val))


