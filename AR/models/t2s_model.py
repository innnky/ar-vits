# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/t2s_model.py
import torch
from tqdm import tqdm

from AR.models.utils import make_pad_mask
from AR.models.utils import topk_sampling
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from AR.modules.wildttstransformer import TTSDecoder

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024
}
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        # assert self.EOS == 1024
        hp = AttrDict()
        hp.__dict__ = config['model']

        self.EOS = hp.n_codes
        self.bert_proj = nn.Linear(1024, hp.hidden_size)
        self.ar_text_embedding = TokenEmbedding(
            hp.hidden_size, hp.phoneset_size, 0)

        self.decoder = TTSDecoder(hp)

        self.ar_accuracy_metric = MulticlassAccuracy(
            hp.n_codes+3,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS, )

    def forward(self, x, x_lens, y, y_lens, bert_feature):
        '''
        x: phoneme_ids
        y: semantic_ids
        '''
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1,2))

        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(torch.int64)
        codes = y.type(torch.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)

        res = self.decoder(y, x, y_mask, x_mask)
        logits = res["logits"].permute(0, 2, 1)

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = F.cross_entropy(logits, targets, reduction='sum')
        acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(self,
              x,
              x_lens,
              prompts,
              bert_feature,
              top_k: int=-100,
              early_stop_num: int=-1,
              temperature: float=1.0):

        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))

        x_mask = make_pad_mask(x_lens)

        y = prompts

        res, alignment = self.decoder.inference_topkp_sampling_batch(x, x_mask, prior=y)

        return res[:, 1:-1]

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(
            y, (0, 1), value=0) + eos_id * F.pad(
                y_mask_int, (0, 1), value=1)
        # 错位
        return targets[:, :-1], targets[:, 1:]
