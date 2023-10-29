import math
import multiprocessing
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

from feature_extractor import cnhubert as content_module
import utils
import logging

from models_vq import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(f, file_path, model,vq_model, device):

    try:
        wav16k, sr = librosa.load(file_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(model, wav_16k_tensor=wav16k)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        f.write(f"{file_path.replace(in_dir, '')}\t{semantic}\n")
        f.flush()
    except:
        print("skip", file_path)

def process_batch(filenames):
    print("Loading hubert for content...")
    process_idx = mp.current_process()._identity
    rank = process_idx[0] if len(process_idx) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    ssl_model = content_module.get_model().to(device)
    hps = utils.get_hparams_from_file("configs/config.json")
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    vq_model.eval()
    utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model,
                                                       None, True)

    print("Loaded hubert.")
    with open(f"dump/semantic_{process_idx[0]}.tsv", "w") as f:
        for filename in tqdm(filenames):
            process_one(f, filename, ssl_model,vq_model, device)

in_dir = "/home/fish/wenetspeech/dataset"

if __name__ == "__main__":
    filenames = glob(f"{in_dir}/**/*.flac", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 8
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()