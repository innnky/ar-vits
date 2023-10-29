# from text.cleaner import clean_text
# text = "所以听到他疯了的消息时，我的第一反应就是假的。他怎么可能会疯呢？他是将「明镜止水」贯彻到极致的人啊。"
# text = "广州的%品牌%有什么$"
#
# phones = clean_text(text, 'zh')
# print(phones)
import os

audiopath = '/Users/xingyijin/Downloads/SSB00120001.wav'
print(os.path.getsize(audiopath)/44100/2)