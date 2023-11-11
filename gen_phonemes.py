import multiprocessing
import os.path
from glob import glob

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from data_conf import data_root
from text.cleaner import clean_text
import numpy as np
from multiprocessing import Pool

out_dir = "dump"
os.makedirs(out_dir, exist_ok=True)
phoneme_path = os.path.join(out_dir, "phoneme.npy")
phone_dict = {}
bert_models = [None] * torch.cuda.device_count()
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")


def get_bert_feature(text, word2ph):
    global bert_models
    rank = multiprocessing.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")

    if bert_models[gpu_id] == None:
        bert_models[gpu_id] = AutoModelForMaskedLM.from_pretrained(
            "hfl/chinese-roberta-wwm-ext-large"
        ).to(device)
        print('loaded bert model at rank', gpu_id)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_models[gpu_id](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T




def process_file(data):
    wav_path, language = data
    if "aidatatang" in wav_path:
        return None
    lab_path = wav_path.replace(".wav", ".lab")
    if os.path.exists(lab_path):
        text = open(lab_path).readline().strip()
        if '{' in text:
            print(f"Error in genshin, {text}")
            return None
        try:
            phones, word2ph, norm_text = clean_text(text, language)
            bert_feature = get_bert_feature(norm_text, word2ph)
            assert bert_feature.shape[-1] == len(phones)
            phones = " ".join(phones)
            torch.save(bert_feature, wav_path.replace(".wav", ".bert.pt"))
            return (wav_path, phones)
        except Exception as e:
            print(f"Error in {wav_path}, {text}", e)
            return None
    else:
        return None
if __name__ == '__main__':

    for language in ["zh"]:
        filenames = glob(f"{data_root}/{language}/**/*.wav", recursive=True)

        # Define the number of processes to use
        num_processes = 8  # You can adjust this as needed
        # multiprocessing.set_start_method("spawn", force=True)

        with Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(process_file, [(f, language) for f in filenames]), total=len(filenames)))

        for result in results:
            if result is not None:
                phone_dict[result[0]] = result[1]

    np.save(phoneme_path, phone_dict)
