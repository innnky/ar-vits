import os.path
from glob import glob
from tqdm import tqdm
from transformers import AutoTokenizer

from data_conf import data_root
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

out_dir = "dump"
os.makedirs(out_dir, exist_ok=True)
phoneme_path = os.path.join(out_dir, "phoneme.npy")
phone_dict = {}

for language in ["zh", 'en']:
    filenames = glob(f"{data_root}/{language}/**/*.wav", recursive=True)


    for filename in tqdm(filenames):
        lab_name = filename.replace(".wav", '.lab')
        if not os.path.exists(lab_name):
            continue
        text = open(lab_name).read().strip()
        text = text.replace("%", '-').replace('￥', ',').replace('^', '')
        phone_dict[filename] = text

np.save(phoneme_path, phone_dict)
