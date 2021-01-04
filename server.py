import ast

import flask
from flask import request, jsonify
import auto_suggest_service as qcs
import preprocess_data as ppd
import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# get suggestions based on input string and character
@app.route('/api/v1/get_suggestions', methods=['GET'])
def get_suggestions():
    # get the value of the string (i.e. ?character=some-value)
    char = request.args.get('character', None)

    with open('data/train_'+str(char)+'.txt', 'r') as file:
        train = [sentence.rstrip() for sentence in file.readlines()]

    with open('data/vocab_'+str(char)+'.txt', 'r') as file:
        vocab = [word.rstrip() for word in file.readlines()]

    # get the value of the string (i.e. ?query_string=some-value)
    input_string = [request.args.get('query_string', None)]
    input_string = pd.DataFrame(input_string, columns=['sentence'])
    tokenized_input = ppd.tokenize_sentences(input_string)
    flat_input = [item for sublist in tokenized_input for item in sublist]

    n_gram_counts_list = []
    for n in range(1, 6):
        print("Computing n-gram counts with n =", n, "...")
        n_model_counts = qcs.count_n_grams(train, n)
        n_gram_counts_list.append(n_model_counts)

    result = qcs.get_suggestions(flat_input, n_gram_counts_list, vocab)
    print(result)
    return jsonify(result)


# Return overall stats for the data
@app.route('/api/v1/get_stats', methods=['GET'])
def get_stats():
    result = ppd.get_stats()
    return jsonify(result)


@app.route('/api/v1/get_perplexity', methods=['GET'])
def get_perplexity():
    # get the value of the string (i.e. ?character=some-value)
    char = request.args.get('character', None)

    with open('data/test_'+str(char)+'.txt', 'r') as file:
        test = [sentence.rstrip() for sentence in file.readlines()]

    test_list = []
    for sentence in test:
        sentence = ast.literal_eval(sentence)
        test_list.append(sentence)

    flat_test_list = [item for sublist in test_list for item in sublist]
    test_set = set(flat_test_list)

    n_gram_counts = qcs.count_n_grams(test, 3)
    n_plus_one_gram_counts = qcs.count_n_grams(test, 4)

    perplexities = dict()
    for sentence in test_list:
        perplexities[str(sentence)] = qcs.calculate_perplexity(
            test_set,
            sentence,
            n_gram_counts,
            n_plus_one_gram_counts)

    return jsonify(perplexities)


if __name__ == '__main__':
    app.run()
