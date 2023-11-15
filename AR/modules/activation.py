# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
import torch
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, softmax_temp=1.0,device=None,dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.k_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.q_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.softmax_temp = softmax_temp

    def reshape(self, x):
        x = x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2).contiguous()
        x = x.view(-1, x.size(2), self.head_dim)
        return x

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, attn_bias=None, past_kv=None):
        batch_size = q.size(0)
        q = self.q_proj(q) * self.head_dim ** -0.5
        if past_kv is not None:
            k, v = torch.cat([past_kv, k], 1), torch.cat([past_kv, v], 1)
        k, v = self.k_proj(k), self.v_proj(v)
        #Reshape for heads (B*nH, T, C)
        q, k, v = self.reshape(q), self.reshape(k), self.reshape(v)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (batch_size * self.nhead, q.size(1), k.size(1))
        if attn_bias is not None:
            assert attn_bias.size() == (self.nhead, q.size(1), k.size(1)), f"Should be {(self.nhead, q.size(1), k.size(1))}. Got {attn_bias.size()}"
            attn_weights = attn_weights + attn_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(batch_size * self.nhead, q.size(1), k.size(1))
        if attn_mask is not None:
            if len(attn_mask.size()) == 2:
                assert attn_mask.size() == (
                q.size(1), k.size(1)), f"Should be {(q.size(1), k.size(1))}. Got {attn_mask.size()}"
                assert attn_mask.dtype == torch.bool
                attn_mask = attn_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1)
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask

            assert attn_mask.size() == (batch_size * self.nhead, q.size(1), k.size(1)), f"Should be {(batch_size * self.nhead, q.size(1), k.size(1))}. Got {attn_mask.size()}, {attn_mask}"
            assert attn_mask.dtype == torch.float
            attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights * self.softmax_temp, dim=-1, dtype=attn_weights.dtype)
        attn_weights_reshaped = attn_weights.view(batch_size, self.nhead, q.size(1), k.size(1))
#        attn_weights = attn_weights_reshaped.view(batch_size * self.nhead, q.size(1), k.size(1))
        attn_probs = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (batch_size * self.nhead, q.size(1), self.head_dim)
        attn_output = attn_output.view(batch_size, self.nhead, q.size(1), self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped
