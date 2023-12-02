import math
import multiprocessing
import os
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

import utils
import logging

from data_conf import data_root
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(f, file_path, model,vq_model, device):

    try:
        # wav16k, sr = librosa.load(file_path, sr=16000)
        # wav16k = torch.from_numpy(wav16k).to(device)
        # ssl_content = content_module.get_content(model, wav_16k_tensor=wav16k)
        ssl_content = torch.load(file_path.replace(".wav", ".ssl.pt")).float().to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        f.write(f"{file_path}\t{semantic}\n")
        f.flush()
    except:
        print("skip", file_path)

def process_batch(filenames):
    print("Loading models ...")
    process_idx = mp.current_process()._identity
    rank = process_idx[0] if len(process_idx) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    # ssl_model = content_module.get_model().to(device)
    hps = utils.get_hparams_from_file("configs/s2-ft.json")
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    vq_model.eval()
    utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model,
                                                       None, True)

    print("Loaded .")
    with torch.no_grad():
        with open(f"dump/semantic_{process_idx[0]}.tsv", "w") as f:
            for filename in tqdm(filenames):
                process_one(f, filename, None ,vq_model, device)

in_dir = data_root

if __name__ == "__main__":
    filenames = glob(f"{in_dir}/**/*.wav", recursive=True)  # [:10]
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

    for p in processes:
        p.join()
    with open(f"dump/semantic.tsv", "w") as f:
        f.write("item_name\tsemantic_audio\n")
        for i in range(num_processes):
            with open(f"dump/semantic_{i+1}.tsv", "r") as f2:
                f.write(f2.read())
            os.remove(f"dump/semantic_{i+1}.tsv")
