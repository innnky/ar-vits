import os.path
from glob import glob
from tqdm import tqdm
from text.cleaner import clean_text
import numpy as np
from multiprocessing import Pool

data_root = '/home/fish/wenetspeech/dataset'
out_dir = "dump"
os.makedirs(out_dir, exist_ok=True)
phoneme_path = os.path.join(out_dir, "phoneme.npy")
phone_dict = {}

def process_file(flac_path):
    lab_path = flac_path.replace(".flac", ".txt")
    if os.path.exists(lab_path):
        text = open(lab_path).readline().strip()
        phones = clean_text(text, 'zh')
        phones = " ".join(phones)
        return (flac_path.replace(data_root, ""), phones)
    else:
        return None

filenames = glob(f"{data_root}/**/*.flac", recursive=True)

# Define the number of processes to use
num_processes = 30  # You can adjust this as needed

with Pool(num_processes) as pool:
    results = list(tqdm(pool.imap(process_file, filenames), total=len(filenames)))

for result in results:
    if result is not None:
        phone_dict[result[0]] = result[1]

np.save(phoneme_path, phone_dict)
