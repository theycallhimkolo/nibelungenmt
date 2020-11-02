import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
from prepare_data import load_clean_sentences


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(lines):
    return max(len(line.split()) for line in lines)


dataset = load_clean_sentences('ger-mhd-both.pkl')
train = load_clean_sentences('ger-mhd-train.pkl')
test = load_clean_sentences('ger-mhd-test.pkl')