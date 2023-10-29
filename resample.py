import os
import shutil
from glob import glob

import librosa
import soundfile
from tqdm import tqdm
from multiprocessing import Pool

def process_wav(wavpath):
    wav, _ = librosa.load(wavpath, sr=tgt_sr)
    soundfile.write(wavpath, wav, tgt_sr)

def get_wav_files(path):
    wav_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

tgt_path = '/home/fish/data_raw'

num_processes = 48  # You can adjust the number of processes as needed
tgt_sr = 32000

if __name__ == "__main__":
    with Pool(num_processes) as pool:
        file_list = get_wav_files(tgt_path)
        list(tqdm(pool.imap(process_wav, file_list), total=len(file_list)))

