import random

import nltk
import pandas as pd

nltk.data.path.append('.')


# 1. Preprocess data
def read_data(file_name):
    data = pd.read_csv(file_name + '.csv',
                       names=['sentence', 'character']) \
        .apply(lambda x: x.astype(str).str.lower())
    cleaned_data = clean_data(data)
    return cleaned_data


def clean_data(data):
    data['sentence'] = data['sentence'].str.replace(r"\(.*\)", "", regex=True)
    data['sentence'] = data['sentence'].str.replace(r"\[.*\]", "", regex=True)
    data['sentence'] = data['sentence'].str.replace("���", "'")
    data['sentence'] = data['sentence'].str.replace(r'\"', "", regex=True)
    data['sentence'] = data['sentence'].str.replace(r'\.', "", regex=True)
    data['sentence'] = data['sentence'].str.replace(r'[^a-zA-Z ]+', '', regex=True)
    return data


def tokenize_sentences_with_character(data, character):
    tokenized_sentences = []
    for sentence in data['sentence'][data['character'] == character]:
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def tokenize_sentences(data):
    tokenized_sentences = []
    for sentence in data['sentence']:
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def split_data(data, seed, character):
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_data = tokenize_sentences_with_character(data, character)

    random.seed(seed)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * 0.8)
    train_data = tokenized_data[0:train_size]
    test_data = tokenized_data[train_size:]

    return train_data, test_data


def preprocess_data(data, count_threshold, seed, character):
    train_data, test_data = split_data(data, seed, character)

    vocabulary = get_words_with_n_plus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)

    return train_data_replaced, test_data_replaced, vocabulary


def get_words_with_n_plus_frequency(tokenized_sentences, count_threshold):
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)
    for word, cnt in word_counts.items():
        if cnt >= count_threshold:
            closed_vocab.append(word)
    return closed_vocab


def count_words(tokenized_sentences):
    word_counts = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1

    return word_counts


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    vocabulary = set(vocabulary)
    replaced_tokenized_sentences = []

    for sentence in tokenized_sentences:

        replaced_sentence = []
        for token in sentence:
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)

        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


def get_stats():
    data = read_data("data/all_lines_the_office")
    char_dict = dict()

    all_characters = data['character'].unique()
    char_dict["num_all_characters"] = len(all_characters)
    char_dict["num_all_lines"] = len(data.index)

    data = data.groupby('character').filter(lambda x: len(x) > 100)
    considered_characters = data['character'].unique()
    char_dict["num_all_considered_characters"] = len(considered_characters)
    for character in considered_characters:
        count_char = len(data.loc[data['character'] == character])
        char_dict["num_" + str(character)+"_lines"] = count_char

    return char_dict


if __name__ == '__main__':
    # Change it here if you want to preprocess the lines of a different character
    # (example: michael, pam, jim, dwight, angela, oscar, stanley, phyllis, etc)
    char = 'ryan'

    # Minimum frequency of words accepted in the vocabulary
    n = 1

    # Read csv file with all lines with the respective character that spoke the line
    all_lines_data = read_data("data/all_lines_the_office")

    # Preprocess data and separate the sentences into train and test along with all words
    # accepted as the vocabulary
    train, test, vocab = preprocess_data(all_lines_data, n, 56, char)

    # Store in files
    with open('data/vocab_'+str(char)+'.txt', 'w') as file:
        for word_ in vocab:
            file.write('%s\n' % word_)

    with open('data/test_'+str(char)+'.txt', 'w') as file:
        for sentence_ in test:
            file.write('%s\n' % sentence_)

    with open('data/train_'+str(char)+'.txt', 'w') as file:
        for sentence_ in train:
            file.write('%s\n' % sentence_)
