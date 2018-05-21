# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def convert_to_words(line):
    words = list()
    word_pairs = [w.strip() for w in line.split(' ')]
    for word_pair in word_pairs:
        if len(word_pair) == 0:
            continue
        if word_pair[0] == '[':
            word_pair = word_pair[1:]
        word_pair = word_pair.split(']/')[0]
        if word_pair.rfind('/') != -1:
            ind = word_pair.rindex('/')
            words.append(word_pair[:ind])
        else:
            words.append(word_pair)
    words = [w.strip() for w in words]
    words = [w for w in words if len(w) > 0]
    return words

def word_tagging_BIEO(words):
    if isinstance(words, str):
        words = words.split(' ')
    chars, tags = list(), list()
    for word in words:
        if len(word) == 1:
            chars.append(word)
            tags.append("O")
        else:
            chars.append(word[0])
            tags.append("B")
            for char in word[1:-1]:
                chars.append(char)
                tags.append("I")
            chars.append(word[-1])
            tags.append("E")
    return chars, tags


def chartags_to_words_BIEO(chars, tags):
    words = list()
    word = ''
    for ch, tag in zip(chars, tags):
        word += ch
        if tag == 'O':
            words.append(word)
            word = ''
        elif tag == 'E':
            words.append(word)
            word = ''
    if len(word) > 0:
        words.append(word)
    return words


def word_tagging_BIO(words):
    if isinstance(words, str):
        words = words.split(' ')
    chars, tags = list(), list()
    for word in words:
        if len(word) == 1:
            chars.append(word)
            tags.append("O")
        else:
            chars.append(word[0])
            tags.append("B")
            for char in word[1:]:
                chars.append(char)
                tags.append("I")
    return chars, tags


def chartags_to_words_BIO(chars, tags):
    words = list()
    word = ''
    for ch, tag in zip(chars, tags):
        if tag == 'O':
            if len(word) > 0:
                words.append(word)
            word = ''
        elif tag == 'B':
            if len(word) > 0:
                words.append(word)
            word = ''
        word += ch
    if len(word) > 0:
        words.append(word)
    return words




