# GPT-VITS

(WIP) text to speech using decoder-only transformer and VITS 
## Note
+ 模型效果未完全验证，不一定会好，请谨慎踩坑，预训练模型还在练
+ 从零训练需要海量数据（至少上千小时？）（类似valle、speartts、soundstorm）数据量少一定不会有好效果。。
+ 由于vits+refenc在zeroshot方向局限性很大，因此本仓库不追求zeroshot，本仓库的目标是，在有一个大的lm的pretrain的情况下，借助自回归lm的力量，希望在对小数据finetune以后能有很好的韵律。
+ 简单更新了一些初步的 [合成samples](https://huggingface.co/innnky/ar-tts-models/tree/main/gpt-vits)
## Todo
+ [x] 在原神数据上训练
+ [x] 收集更多中文开源数据训练（预计600H左右）训练并放出pretrain（x） --> out-of-distribution文本效果很差，例如读文言文 并且长句效果不好, 会抽风
  + [ ] 添加word level bert 并repeat到phoneme level改善out-of-distribution效果
  + [ ] 将同一spk的数据多条合并为一条音频 提高平均数据时长 改善长句合成效果稳定性
  + [ ] 更换为RoPE相对位置编码改善长句合成效果稳定性？
+ [ ] 编写finetune相关代码，增加sid支持
+ [ ] 优化日语和英语文本前端，收集更多日、英数据（预计每种语言600H）训练并放出pretrain

## structure
![structure.png](resources%2Fstructure.png)

+ decoder only text2semantic from [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
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
