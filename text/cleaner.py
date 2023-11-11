from text import chinese, japanese, cleaned_text_to_sequence, symbols, english

language_module_map = {
    'zh': chinese,
    "ja": japanese,
    'en': english
}
special = [
    ('%', 'zh', "SP"),
    ('￥', 'zh', "SP2"),
    ('^', 'zh', "SP3")
]
def clean_text(text, language):
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, word2ph = language_module.g2p(norm_text)
    assert len(phones) == sum(word2ph)
    assert len(norm_text) == len(word2ph)

    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, word2ph = language_module.g2p(norm_text)
    assert len(phones) == sum(word2ph)
    assert len(norm_text) == len(word2ph)

    new_ph = []
    for ph in phones:
        assert ph in symbols
        if ph == ',':
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, word2ph, norm_text

def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)

if __name__ == '__main__':
    print(clean_text("你好%啊啊啊额、还是到付红四方。", 'zh'))


