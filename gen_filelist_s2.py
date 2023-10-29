from glob import glob
from random import shuffle
from data_conf import data_root

filenames = glob(f"{data_root}/**/*.wav", recursive=True)  # [:10]

shuffle(filenames)
val_num = 8
train = filenames[:-val_num]
val = filenames[-val_num:]
train.sort()
val.sort()

with open('dump/s2_train_files.list', 'w') as f:
    f.write('\n'.join(train))
with open('dump/s2_val_files.list', 'w') as f:
    f.write('\n'.join(val))


