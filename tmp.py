import json

import matplotlib.pyplot as plt
import torch

from AR.modules.wildttstransformer import TTSDecoder


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

hp = AttrDict()
config_path = 'configs/s1.json'
with open(config_path, 'r') as f:
    argdict = json.load(f)
    hp.__dict__ = argdict

decoder = TTSDecoder(hp)

q = torch.randint(0, 100, (1, 31))
q_mask = torch.zeros(1, 31).bool()
phone = torch.randn(1, 13, 768)
spkr = torch.randn(1, 768)
phone_mask = torch.zeros(1, 13).bool()

res = decoder.inference_topkp_sampling_batch(phone, phone_mask, )
