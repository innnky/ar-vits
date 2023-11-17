import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .transformers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, CrossAttnOnlyLayer, AlibiPostionEmbedding
from .transducer import Transducer
import numpy as np
import statistics
from AR.models.utils import topk_sampling

class TTSDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                [TransformerEncoderLayer(hp) for i in range(hp.enc_nlayers)]
            )
        )
        self.decoder = TransformerDecoder(
            nn.ModuleList(
                [TransformerDecoderLayer(hp, with_cross_attention=False) for i in range(hp.dec_nlayers)]
            )
        )
        self.aligner = CrossAttnOnlyLayer(hp)
        self.layer_norm_phone = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.transducer = Transducer(hp)
        self.alibi = AlibiPostionEmbedding(hp.nheads, 3000)
        self.layer_norm = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.tgt_mask = (torch.tril(torch.ones(3000, 3000), diagonal=0) == 0)

    def forward(self, q, phone, q_mask, phone_mask):
        #Fused phone + speaker
        ex_phone_mask = phone_mask
        phone = self.layer_norm_phone(phone)
        phone_alibi = self.alibi(phone)
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=ex_phone_mask)

        #Run decoder
        q_input = q
        q = self.transducer.encode(q)
        q = self.layer_norm(q)
        tgt_len = q.size(1)
        tgt_mask = self.tgt_mask[: tgt_len, : tgt_len].to(q.device)
        audio_alibi = self.alibi(q)
        output, _, dec_attn, _ = self.decoder(q, memory=None,
                                              tgt_mask=tgt_mask,
                                              attn_bias=audio_alibi,
                                              tgt_key_padding_mask=q_mask,
                                              memory_key_padding_mask=None)
        output, alignment = self.aligner(output, phone, tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=q_mask, memory_key_padding_mask=phone_mask)
        audio_output = self.transducer.decode(output)
        return {
            'logits': audio_output,
            'alignment': alignment,
            'decoder_attention': dec_attn,
            'encoder_attention': enc_attn
        }

    def encode_phone(self, phone, phone_mask):
        phone = self.layer_norm_phone(phone)
        phone_alibi = self.alibi(phone)
        phone, enc_attn = self.encoder(phone, mask=None, attn_bias=phone_alibi, src_key_padding_mask=phone_mask)
        return phone

    def inference_topkp_sampling_batch(self, phone, phone_mask, prior=None, output_alignment=False):
        final_outputs = [prior]
        batch_size = phone.size(0)
        inp = self.layer_norm(self.transducer.start_token(phone.device)) #1, 1, C
        inp = inp.expand(batch_size, -1, -1) #N, 1, C
        prior_size = 0
        if prior is not None:
            prior = self.transducer.encode(prior)
            prior = self.layer_norm(prior)
            prior_size = prior.size(1)
            inp = torch.cat([inp, prior], 1)
        phone = self.encode_phone(phone, phone_mask)
        tgt_mask = self.tgt_mask[:inp.size(1), :inp.size(1)].to(inp.device)
        inps = inp

        #Decode
        past_kvs1, past_kv_cross, past_kvs2 = None, None, None
        audio_alibi = self.alibi(inp)
        back_map = torch.zeros([batch_size, 1], device=phone.device, dtype=torch.long)
        length_counter = torch.zeros([batch_size], device=phone.device, dtype=torch.long)
        real_phone_lengths = (~phone_mask).long().sum(-1) #N,
        assert batch_size == 1
        alignment = torch.zeros((1, self.hp.max_output_length, self.hp.max_output_length), device=phone.device)
        for i in tqdm(range(self.hp.max_output_length)):
            cond, _, _, new_1 = self.decoder(inp, memory=None, attn_bias=audio_alibi, tgt_mask=tgt_mask, past_kvs=past_kvs1)
            #Only feed in the current frame and the next frame attending!
            t_length, c_length = phone.size(1), phone.size(2) # T, C
            selected_phone = phone.reshape(-1, c_length) #N*T, C
            index_map = torch.arange(self.hp.phone_context_window, device=phone.device)
            index_map = back_map[:, -1:] + index_map.repeat(batch_size, 1)
            add = torch.arange(batch_size, device=index_map.device).unsqueeze(1) #N, 1
            index_map = index_map + add * t_length
            index_map = index_map.reshape(-1) #N * 3
            selected_phone = selected_phone[index_map].reshape(batch_size, self.hp.phone_context_window, c_length) #N*3, C
            #Mask for the starting phones
            phone_mask = torch.arange(self.hp.phone_context_window, device=phone.device).repeat(batch_size, 1)
            phone_mask = (phone_mask <= (back_map[:, -1:] + 1).expand(-1, self.hp.phone_context_window))
            phone_mask = ~phone_mask
            cond, _align = self.aligner(cond, selected_phone, tgt_mask=tgt_mask, memory_key_padding_mask=phone_mask)
            cond = cond[:, -1].unsqueeze(1) #N, 1, C
            #Run sub-decoder inference
            logits = self.transducer.decode(cond).squeeze(0)
            samples = topk_sampling(
                logits, top_k=-100, top_p=1.0, temperature=1.0)
            output = samples
            final_outputs.append(output)
            if self.transducer.is_end_token_batch(output):
                break

            news = [inps] + new_1

            #Update args
            tgt_mask = self.tgt_mask[i+2+prior_size, :i+2+prior_size].to(phone.device).unsqueeze(0)
            audio_alibi = self.alibi(tgt_mask)[:, -1].unsqueeze(1)
            audio_alibi[:, :, 0] = 0
            alignment[:, i, back_map[0, -1]: back_map[0, -1]+self.hp.phone_context_window] = _align[:, 0, -1].unsqueeze(0)
            next_idx = (_align[:, 0, -1, 0] < (1 / self.hp.phone_context_window)).long()
            next_idx[length_counter >= self.hp.length_penalty_max_length] = 1
            new_bk = torch.minimum(back_map[:, -1] + next_idx, real_phone_lengths - self.hp.phone_context_window)
            back_map = torch.cat([back_map, new_bk.unsqueeze(1)], 1)
            length_counter[next_idx == 0] += 1
            length_counter[next_idx != 0] = 0
            if i == 0:
                past_kvs1 = news[:self.hp.dec_nlayers]
            else:
                news = [x[:, -1:] for x in news]
                for ii, (p, n) in enumerate(zip(past_kvs1, news[:self.hp.dec_nlayers])):
                    past_kvs1[ii] = torch.cat([p, n], 1)



            inp = self.transducer.encode(output)
            inp = self.layer_norm(inp)
            inps = torch.cat([inps, inp], 1)
        return torch.cat(final_outputs, 1), alignment[:, :i, :phone.size(1)]
