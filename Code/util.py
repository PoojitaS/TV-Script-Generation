import os
import torch
import pickle
import random
from nltk.corpus import stopwords, words

SPECIAL_WORDS = {'PADDING': '<PAD>'}

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    s = set(stopwords.words('english'))

    #Reduce freuqency of stop words by adding them only 50% of the time.
    int_text = [vocab_to_int[word] for word in text if (word not in s) or (word in s and random.random() < 0.5)]

    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))

def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    path = os.getcwd()  +"/"+ save_filename
    torch.save(decoder, path)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    path = os.getcwd() +"/"+ save_filename
    return torch.load(path)
