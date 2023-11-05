# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/t2s_dataset.py
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from text import cleaned_text_to_sequence


def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (
                ndim - axis - 1)
        padded_seq = np.pad(
            seq, padding, mode='constant', constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(self,
                 phoneme_path: str,
                 semantic_path: str,
                 max_sample: int = None,
                 max_sec: int = 100,
                 pad_val: int = 1024,
                 # min value of phoneme/sec
                 min_ps_ratio: int = 3,
                 # max value of phoneme/sec
                 max_ps_ratio: int = 25) -> None:
        super().__init__()

        self.semantic_data = pd.read_csv(semantic_path, delimiter='\t')
        # get dict
        self.phoneme_data = np.load(phoneme_path, allow_pickle=True).item()
        # pad for semantic tokens
        self.PAD: int = pad_val
        self.hz = 25
        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        if max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]

        # {idx: (semantic, phoneme)}
        # semantic list, phoneme list
        self.semantic_phoneme = []
        self.item_names = []

        self.inited = False

        if not self.inited:
            # 调用初始化函数
            self.init_batch()
            self.inited = True
            del self.semantic_data
            del self.phoneme_data
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")


    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("phoneme_data_len:", phoneme_data_len)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(semantic_data_len):
            # 先依次遍历
            # get str
            item_name = self.semantic_data['item_name'][i]
            try:
                text = self.phoneme_data[item_name]
            except Exception:
                # print(f"{item_name} not in self.phoneme_data !")
                num_not_in += 1
                continue

            semantic_str = self.semantic_data['semantic_audio'][i]
            # get token list
            semantic_ids = [int(idx) for idx in semantic_str.split(' ')]
            # (T), 是否需要变成 (1, T) -> 不需要，因为需要求 len
            # 过滤掉太长的样本
            if len(semantic_ids) > self.max_sec * self.hz:
                num_deleted_bigger += 1
                continue

            if len(semantic_ids) > 1000:
                num_deleted_bigger += 1
                continue


            self.semantic_phoneme.append((semantic_ids, text))
            idx += 1
            self.item_names.append(item_name)
        if num_not_in > 0:
            print(f"there are {num_not_in} semantic datas not in phoneme datas")
        if num_deleted_bigger > 0:
            print(
                f"deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
            )
        if num_deleted_ps > 0:
            # 4702 for LibriTTS, LirbriTTS 是标注数据, 是否需要筛？=> 需要，有值为 100 的极端值
            print(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
            )
        # 345410 for LibriTTS
        print("dataset.__len__():", self.__len__())

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, text = self.semantic_phoneme[idx]
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)
        return {
            'idx': idx,
            'text': text,
            'semantic_ids': semantic_ids,
            'semantic_ids_len': semantic_ids_len
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        texts: List[str] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []

        for item in examples:
            sample_index.append(item["idx"])
            texts.append(item["text"])
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            semantic_ids_lens.append(item["semantic_ids_len"])

        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        semantic_ids = torch.tensor(semantic_ids)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, return_length=True)

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length) 
            "input_ids": inputs['input_ids'],
            "token_type_ids": inputs['token_type_ids'],
            "attention_mask": inputs['attention_mask'],
            # torch.Tensor (B)
            "length": inputs['length'],
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
        }


if __name__ == '__main__':
    root_dir = '/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm/dump_libritts/train/'
    dataset = Text2SemanticDataset(
        phoneme_path=root_dir + 'phonemes.npy',
        semantic_path=root_dir + 'semantic_token.tsv')

    batch_size = 12
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        shuffle=False)
    # for i, batch in enumerate(dataloader):
    #     if i == 0:
    #         print('batch["ids"]:', batch["ids"])
    #         print('batch["phoneme_ids"]:', batch["phoneme_ids"],
    #               batch["phoneme_ids"].shape)
    #         print('batch["phoneme_ids_len"]:', batch["phoneme_ids_len"],
    #               batch["phoneme_ids_len"].shape)
    #         print('batch["semantic_ids"]:', batch["semantic_ids"],
    #               batch["semantic_ids"].shape)
    #         print('batch["semantic_ids_len"]:', batch["semantic_ids_len"],
    #               batch["semantic_ids_len"].shape)
