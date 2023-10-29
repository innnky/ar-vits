import math
import multiprocessing
import os
import argparse
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

import cnhubert as content_module
import utils
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(file_path, model, device):

    ssl_path = file_path.replace(".flac", ".hubert.pt")
    # try:
    #     torch.load(ssl_path)
    # except:
    try:
        wav16k, sr = librosa.load(file_path, sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        ssl_content = content_module.get_content(model, wav_16k_tensor=wav16k)
        torch.save(ssl_content.cpu().half(), ssl_path)
        del ssl_content
        del wav16k
    except:
        print("skip", file_path)

def process_batch(filenames):
    print("Loading hubert for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)
    ssl_model = content_module.get_model().to(device)
    print("Loaded hubert.")
    for filename in tqdm(filenames):
        process_one(filename, ssl_model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="/home/fish/wenetspeech/dataset_vq", help="path to input dir"
    )
    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/*.flac", recursive=True)  # [:10]
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