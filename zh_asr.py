import os
from glob import glob

import librosa

from data_conf import data_root

all_wavs = glob(f"{data_root}/**/*.wav", recursive=True)
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

dur_symbol_map = [
    (80, ""),
    (160, "-"),
    (500, "，"),
    (1000, "。")
]
def get_dur_symbol(dur):
    for up, symbol in dur_symbol_map:
        if dur <= up:
            return symbol
    return "..."


inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    model_revision="v1.2.4")
for wav_path in all_wavs:
    wav, sr = librosa.load(wav_path, sr=16000)
    result = inference_pipeline(audio_in=wav)
    texts = result['text_postprocessed'].split(" ")
    time_stamp = result['time_stamp']
    total_end = int(wav.shape[0] / sr * 1000)


    last_time = time_stamp[0][0]

    lab_text = ''
    for i in range(len(texts)):
        st, ed = time_stamp[i][0], time_stamp[i][1]
        if st != last_time:
            lab_text += get_dur_symbol(st - last_time)
        lab_text += texts[i]
        last_time = ed
    if total_end != last_time:
        lab_text += get_dur_symbol(total_end - last_time)


    out_lab_name = wav_path.replace('.wav', '.lab')
    with open(out_lab_name, 'w') as f:
        f.write(lab_text)




