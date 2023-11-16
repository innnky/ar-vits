import json

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

decoder = TTSDecoder(hp, 1000)

q = torch.randint(0, 100, (2, 31))
q_len = torch.LongTensor([31, 31])
q_mask = torch.ones(2, 31).bool()
phone = torch.randn(2, 13, 768)
spkr = torch.randn(2, 768)
phone_mask = torch.ones(2, 13).bool()

res = decoder.forward(q, phone, q_mask, phone_mask, )