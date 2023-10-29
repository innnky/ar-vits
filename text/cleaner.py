from text import chinese, japanese, cleaned_text_to_sequence, symbols, english

language_module_map = {
    'zh': chinese,
    "ja": japanese,
    'en': english
}

def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)

    for ph in phones:
        assert ph in symbols
    return phones

def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)

if __name__ == '__main__':
    print(clean_text("你好，啊啊啊额、还是到付红四方。"))


