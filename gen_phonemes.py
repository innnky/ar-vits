import os.path
from glob import glob
from tqdm import tqdm

from data_conf import data_root
from text.cleaner import clean_text
import numpy as np
from multiprocessing import Pool

out_dir = "dump"
os.makedirs(out_dir, exist_ok=True)
phoneme_path = os.path.join(out_dir, "phoneme.npy")
phone_dict = {}

def process_file(wav_path, language):
    lab_path = wav_path.replace(".wav", ".lab")
    if os.path.exists(lab_path):
        text = open(lab_path).readline().strip()
        phones = clean_text(text, language)
        phones = " ".join(phones)
        return (wav_path, phones)
    else:
        return None
for language in ["zh", 'en', 'ja']:
    filenames = glob(f"{data_root}/{language}/**/*.wav", recursive=True)

    # Define the number of processes to use
    num_processes = 30  # You can adjust this as needed

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, [(f, language) for f in filenames]), total=len(filenames)))

    for result in results:
        if result is not None:
            phone_dict[result[0]] = result[1]

np.save(phoneme_path, phone_dict)
