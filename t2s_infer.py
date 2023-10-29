import os

import numpy as np
import pandas as pd
import torch

from t2s.t2s_up import TSARTransformer
from text import cleaned_text_to_sequence
from text.cleaner import clean_text


# text to semantic
import argparse
import os
import re
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from infer import encode_semantic_from_wav16k_numpy
from text import cleaned_text_to_sequence
from text.cleaner import text_to_sequence, clean_text


text = "当然,不同问题之间错综复杂,对应的结论也有冲突.所以我想要的是'平衡',也就是在所有问题中找到一个'最优解'."
# text = "当然,不同问题之间错综复杂,对应的结论也有冲突.所以我想要的是'平衡'。"
text = "幸运的是，此次事故并未造成人员伤亡，但两辆车均受到了不同程度的损伤。事故发生后，许多网友对这名女子的驾驶行为表示了强烈的不解和担忧。同时，也有网友表示，这种行为不仅危害了自己和他人的生命安全，还可能对其他道路使用者造成恐慌和困扰。"
# text = "皆さん、こんにちは、私は派蒙です。今日はみんなが見たいものをください。"
prompt_text = "万一他很崇拜我们呢?嘿嘿,"
prompt_wav_path = "/home/fish/genshin_data/zh/派蒙/vo_DQAQ003_1_paimon_06.wav"

def text2phoneid(text, lang='zh'):
    phones = clean_text(text, lang)
    print(phones)
    return cleaned_text_to_sequence(phones)


semantic_data = pd.read_csv('dump/semantic.tsv', delimiter='\t')


phones = text2phoneid(text)
prompt_phones = text2phoneid(prompt_text)
prompt_semantic = semantic_data[semantic_data['item_name'] == prompt_wav_path]['semantic_audio'].values[0]
prompt_semantic = torch.LongTensor([int(idx) for idx in prompt_semantic.split(' ')])

print(prompt_semantic)
n_semantic = 1024
device = 'cpu'
config = load_yaml_config("configs/default.yaml")
ckpt_path = 'logs/ar/ckpt/epoch=34-step=28560.ckpt'

hz = 50
max_sec = config['data']['max_sec']

# get models
t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
    checkpoint_path=ckpt_path, config=config, map_location=device)
t2s_model.to(device)
t2s_model.eval()

total = sum([param.nelement() for param in t2s_model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))


all_phoneme_ids = torch.LongTensor(prompt_phones+phones).to(device).unsqueeze(0)
print(all_phoneme_ids.shape)
all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
prompt = prompt_semantic.unsqueeze(0).to(device)
st = time.time()
with torch.no_grad():
    pred_semantic = t2s_model.model.infer(
        all_phoneme_ids,
        all_phoneme_len,
        prompt,
        top_k=config['inference']['top_k'],
        early_stop_num=hz * max_sec)

print(f'{time.time() - st} sec used in T2S')

torch.save(pred_semantic.squeeze(0).squeeze(0), 'pred_semantic.pt')

phones = " ".join([str(i) for i in prompt_phones+phones])

os.system(f"python infer.py '{phones}'")