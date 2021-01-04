import ast

import nltk
import numpy as np

nltk.data.path.append('.')


# Methods to develop n-gram based language models
def count_n_grams(data, n_size, start_token='<s>', end_token='<e>'):
    n_grams = {}

    for sentence in data:
        sentence = ast.literal_eval(sentence)
        # prepend start token n times, and  append <e> one time
        sentence = [start_token] * n_size + sentence + [end_token]
        sentence = tuple(sentence)

        m = len(sentence) if n_size == 1 else len(sentence) - 1
        for i in range(m):

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i + n_size]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams.keys():

                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1

    return n_grams


# Use perplexity to evaluate
def calculate_perplexity(unique_words, sentence, n_gram_counts, n_plus1_gram_counts):
    n_size = len(list(n_gram_counts.keys())[0])

    # prepend <s> and append <e>
    sentence = ["<s>"] * n_size + sentence + ["<e>"]
    sentence = tuple(sentence)

    N = len(sentence)

    # The variable p will hold the product
    # that is calculated inside the n-root
    product_pi = 1.0

    for t in range(n_size, N):
        # get the n-gram preceding the word at position t
        n_gram = sentence[t - n_size:t]
        word = sentence[t]

        # Estimate the probability of the word given the n-gram
        # using the n-gram counts, n-plus1-gram counts,
        # vocabulary size, and smoothing constant
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, len(unique_words), k=1)

        # Update the product of the probabilities
        # This 'product_pi' is a cumulative product
        # of the (1/P) factors that are calculated in the loop
        product_pi *= np.math.log(1 / probability)

    # Take the Nth root of the product
    perplexity = product_pi ** (1 / float(N))

    return perplexity


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0

    # Calculate the denominator using the count of the previous n gram
    # and apply k-smoothing
    denominator = previous_n_gram_count + k * vocabulary_size
    n_plus1_gram = previous_n_gram + (word,)
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

    # Define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count + k

    probability = numerator / denominator

    return probability


# System to get suggestions
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k)
        suggestions.append(suggestion)
    return suggestions


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    n = len(list(n_gram_counts.keys())[0])

    previous_n_gram = previous_tokens[-n:]

    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    suggestion = None
    max_prob = 0

    for word, prob in probabilities.items():
        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob:
            suggestion = word
            max_prob = prob

    return suggestion, max_prob


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    previous_n_gram = tuple(previous_n_gram)

    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities
