import time

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

import utils
import torch.nn as nn

class CNHubert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-base")
    def forward(self, x):
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats



def get_model():
    model = CNHubert()
    model.eval()
    return model

def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1,2)


if __name__ == '__main__':
    model = get_model()
    src_path = "/Users/Shared/原音频2.wav"
    wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000)
    model = model
    wav_16k_tensor = wav_16k_tensor
    feats = get_content(model,wav_16k_tensor)
    print(feats.shape)

