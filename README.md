# AutoRegressive-VITS

> MQTTS branch

(WIP) text to speech using autoregressive transformer and VITS 
## Note
+ 模型效果未完全验证，不一定会好，请谨慎踩坑，预训练模型还在练
+ 从零训练需要海量数据（至少上千小时？）（类似valle、speartts、soundstorm）数据量少一定不会有好效果。。
+ 由于vits+refenc在zeroshot方向局限性很大，因此本仓库不追求zeroshot，本仓库的目标是，在有一个大的lm的pretrain的情况下，借助自回归lm的力量，希望在对小数据finetune以后能有很好的韵律。
+ 简单更新了一些初步的 [合成samples](https://huggingface.co/innnky/ar-tts-models/tree/main/gpt-vits)

## structure
![structure.png](resources%2Fstructure.png)

+ autoregressive mqtts transformer from [MQTTS](https://github.com/b04901014/MQTTS)
+ VITS from [VITS](https://github.com/jaywalnut310/vits)
+ reference encoder from [TransferTTS](https://github.com/hcy71o/TransferTTS)

## Training pipeline
1. jointly train S2 vits decoder and quantizer
2. extract semantic tokens
3. train S1 text to semantic

## vits S2 training
+ resample.py
+ gen_phonemes.py
+ extract_ssl_s2.py
+ gen_filelist_s2.py
+ train_s2.py

## gpt S1 training
+ extract_vq_s1.py
+ gen_filelist_s1.py
+ train_s1.py

## Inference
+ s1_infer.py/s2_infer.py (work in progress)

## Pretrained models
+ work in progress