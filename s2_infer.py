import sys

import librosa
import soundfile
import torch

import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
# from feature_extractor import cnhubert as content_module

vits_model_cache = None


def _load_model(device="cuda"):
    global vits_model_cache
    if vits_model_cache is not None:
        return vits_model_cache
    hps = utils.get_hparams_from_file("configs/s2.json")
    model_dir = hps.s2_ckpt_dir
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)

    utils.load_checkpoint(utils.latest_checkpoint_path(model_dir, "G_*.pth"), net_g,
                          None, True)
    net_g.eval()
    vits_model_cache = (hps, net_g)
    return hps, net_g


def get_spepc(hps, filename):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, hps.data.sampling_rate))
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False)
    return spec


def decode_to_file(codes,phonemes, save_path, refer_path, transform='valle'):
    device = codes.device
    hps, net_g = _load_model(device=device)
    if transform=='valle':
        codes = codes.transpose(0, 1).unsqueeze(1)
    else:
        codes = codes.transpose(0, 1)
    refer = get_spepc(hps, refer_path).to(device)
    audio = net_g.decode(codes,phonemes, refer).detach().cpu().numpy()[0, 0]
    soundfile.write(save_path, audio, hps.data.sampling_rate)


def encode_from_file(path, device='cpu'):
    hps, net_g = _load_model(device=device)
    content_model = content_module.get_model().to(device)
    wav16k, sr = librosa.load(path, sr=16000)
    with torch.no_grad():
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(content_model, wav_16k_tensor=wav16k)
        codes = net_g.extract_latent(ssl_content)
    return codes.cpu()

def encode_semantic_from_wav16k_numpy(wav16k, device='cpu'):
    hps, net_g = _load_model(device=device)
    content_model = content_module.get_model().to(device)
    with torch.no_grad():
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(content_model, wav_16k_tensor=wav16k)
        codes = net_g.extract_latent(ssl_content)
    return codes[0, :1, :]

if __name__ == '__main__':
    codes_path = "pred_semantic.pt"
    refer_path = "/home/fish/genshin_data/zh/派蒙/vo_DQAQ003_1_paimon_06.wav"
    # src_path = "dataset/PaiMeng/vo_DQAQ003_1_paimon_06.wav"
    device = 'cpu'
    # codes = encode_from_file(src_path, device=device)
    codes = torch.load(codes_path).unsqueeze(0).unsqueeze(0)
    print('argv', sys.argv[1])
    phonemes = torch.LongTensor([int(i) for i in sys.argv[1].split(" ")]).unsqueeze(0)
    print(codes.shape)
    print("phonemes", phonemes)

    decode_to_file(codes, phonemes,"tmp.wav", refer_path, transform="raw")