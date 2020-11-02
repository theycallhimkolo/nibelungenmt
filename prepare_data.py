import string
from pickle import dump, load
from unicodedata import normalize
import numpy as np
import re
from numpy.random import shuffle


def merge_translation():
    with open("data/mhd.txt", "r", encoding="utf-8") as mhd:
        with open("data/ger.txt", "r", encoding="utf-8") as ger:
            with open("data/data.txt", "w", encoding="utf-8") as data:
                for l1, l2 in zip(mhd, ger):
                    data.write(l1.strip() + "\t" + l2.strip() + "\n")


def load_doc(filename):
    file = open("data/" + filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def load_clean_sentences(filename):
    return load(open("data/" + filename, 'rb'))


def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


def save_clean_data(sentences, filename):
    dump(sentences, open("data/" + filename, 'wb'))
    print('Saved: %s' % filename)

def clean_pairs(lines):
    cleaned = []
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [re_print.sub('', w) for w in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)


if __name__ == "__main__":
    merge_translation()

    doc = load_doc("data.txt")
    pairs = to_pairs(doc)
    clean_pairs = clean_pairs(pairs)
    save_clean_data(clean_pairs, 'ger-mhd.pkl')

    raw_dataset = load_clean_sentences('ger-mhd.pkl')

    n_sentences = 150
    dataset = raw_dataset[:n_sentences, :]
    print(dataset)
    shuffle(dataset)
    # split into train/test
    train, test = dataset[:149], dataset[149:]
    # save
    save_clean_data(dataset, 'ger-mhd-both.pkl')
    save_clean_data(train, 'ger-mhd-train.pkl')
    save_clean_data(test, 'ger-mhd-test.pkl')