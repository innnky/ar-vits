import math
import multiprocessing
import argparse
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

import utils
from data_conf import data_root
from feature_extractor import content_module_map
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(file_path, model, device, content_module):

    ssl_path = file_path.replace(".wav", ".ssl.pt")
    try:
        wav16k, sr = librosa.load(file_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(model, wav_16k_tensor=wav16k)
        torch.save(ssl_content.cpu().half(), ssl_path)
        del ssl_content
        del wav16k
    except:
        print("skip", file_path)

def process_batch(filenames, content_module):
    print("Loading hubert for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    ssl_model = content_module.get_model().to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model, device, content_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="configs/s2.json", help="path to config"
    )
    args = parser.parse_args()
    filenames = glob(f"{data_root}/**/*.wav", recursive=True)  # [:10]
    hps = utils.get_hparams_from_file(args.config)
    content_module = content_module_map[hps.content_module]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 8
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,content_module)) for chunk in chunks
    ]
    for p in processes:
        p.start()