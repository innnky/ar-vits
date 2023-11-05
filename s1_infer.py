from pathlib import Path

import pandas as pd
import os
import time
import torch
from transformers import AutoTokenizer

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils import get_newest_ckpt
from AR.utils.io import load_yaml_config
from text import cleaned_text_to_sequence
from text.cleaner import text_to_sequence, clean_text

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

text = "当然,不同问题之间错综复杂,对应的结论也有冲突.所以我想要的是'平衡',也就是在所有问题中找到一个'最优解'."
text = "当然,不同问题之间错综复杂,对应的结论也有冲突."
# text = "幸运的是，此次事故并未造成人员伤亡，但两辆车均受到了不同程度的损伤。事故发生后，许多网友对这名女子的驾驶行为表示了强烈的不解和担忧。同时，也有网友表示，这种行为不仅危害了自己和他人的生命安全，还可能对其他道路使用者造成恐慌和困扰。"
# text = "幸运的是，此次事故并未造成人员伤亡，但两辆车均受到了不同程度的损伤。"
# text = "皆さん、こんにちは、私は道理を言って、今日はみんなが見たいものを注文します。"
# text = "Hello everyone, I mean the point. Today, let's have some things that everyone wants to see."
# text = "子曰：“学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？"
# text = "你说的对，但是原神是由米哈游自主研发的一款全新开放世界冒险游戏。"
# text="先帝创业未半，而中道崩殂；今天下三分，益州疲弊，此诚危急存亡之秋也！然侍卫之臣，不懈于内；忠志之士，忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"
prompt_text = "万一他很崇拜我们呢?嘿嘿,"
prompt_wav_path = "/home/fish/genshin_data/zh/派蒙/vo_DQAQ003_1_paimon_06.wav"


semantic_data = pd.read_csv('dump/semantic.tsv', delimiter='\t')

all_text = prompt_text + text

all_phones = clean_text(all_text, "zh")
all_phones_ids = cleaned_text_to_sequence(all_phones)
inputs = tokenizer([text], return_tensors="pt", padding=True, return_length=True)

prompt_semantic = semantic_data[semantic_data['item_name'] == prompt_wav_path]['semantic_audio'].values[0]
prompt_semantic = torch.LongTensor([int(idx) for idx in prompt_semantic.split(' ')])

print(prompt_semantic)
n_semantic = 1024
device = 'cpu'
config = load_yaml_config("configs/s1.yaml")

output_dir = Path('logs/s1-bert')
ckpt_dir = output_dir / 'ckpt'
newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
ckpt_path = ckpt_dir / newest_ckpt_name
print("ckpt_path:", ckpt_path)
hz = 50
max_sec = config['data']['max_sec']

# get models
t2s_model = Text2SemanticLightningModule.load_from_checkpoint(
    checkpoint_path=ckpt_path, config=config, map_location=device)
t2s_model.to(device)
t2s_model.eval()

total = sum([param.nelement() for param in t2s_model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))


prompt = prompt_semantic.unsqueeze(0).to(device)
st = time.time()
with torch.no_grad():
    pred_semantic = t2s_model.model.infer(
        **inputs,
        prompts=prompt,
        top_k=config['inference']['top_k'],
        early_stop_num=hz * max_sec)

print(f'{time.time() - st} sec used in T2S')

torch.save(pred_semantic.squeeze(0).squeeze(0), 'pred_semantic.pt')

phones = " ".join([str(i) for i in all_phones_ids])

os.system(f"python s2_infer.py '{phones}'")