import math, csv, os, pickle
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

feature_list = [
    ["might", "could", "can", "would", "may"],
    ["should", "ought", "need", "shall", "will", "must"],
    ["if"],
    ["no", "not", "neither", "nor", "never"],
    ["therefore", "furthermore", "consequently", "thus", "as", "subsequently", "eventually", "hence"],
    ["till", "until", "despite", "inspite", "though", "although"],
    ["but", "however", "nevertheless", "otherwise", "yet", "still", "nonetheless"],
    ["i", "we", "me", "us", "my", "mine", "our", "ours"],
    ["you, your, yours"],
    ["he", "she", "him", "her", "his", "it", "its", "hers", "they", "them", "their", "theirs"],
    ["DT"],
    ["WDT", "WP", "WP$", "WRB", "?"],
    ["JJ", "JJR", "JJS"],
    ["RB", "RBR", "RBS"],
    ["NNP", "NNPS"]
]


idx_dict = {
    "Strong modals": 0,
    "Weak modals": 1,
    "Conditionals": 2,
    "Negation": 3,
    "Inferential Conjunctions": 4,
    "Contrasting Conjunctions": 5,
    "Following Conjunctions": 6,
    "First Person": 7,
    "Second Person": 8,
    "Third Person": 9,
    "Determiner": 10,
    "QS": 11,
    "Adjectives": 12,
    "Adverbs": 13,
    "Proper Nouns": 14
}


def save_data(path, py_object):
    with open(path, 'wb') as f:
        pickle.dump(py_object, f)


# Load data from existing pickle
def load_data(pickle_in):
    with open(pickle_in, 'rb') as f:
        contents = pickle.load(f)
    return contents


def divide_batches(fin, n_batches, size):
    print("Divide", fin, "into", n_batches, "batches.")
    tsvin = open(fin + ".tsv", mode='rt')
    tsvin_writer = csv.reader(tsvin, delimiter='\t')
    dir = fin + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    line_per_file = math.ceil(float(size) / n_batches)
    n_out = 1
    n_line = 1
    csvout = open(dir + fin + "_" + str(n_out) + ".csv", "wt")
    csvout_writer = csv.writer(csvout, delimiter=',', lineterminator='\n')
    for row in tsvin_writer:
        print(row)
        if n_line > line_per_file:
            n_out += 1
            csvout.close()
            csvout = open(dir + fin + "_" + str(n_out) + ".csv", "wt")
            csvout_writer = csv.writer(csvout, delimiter=',', lineterminator='\n')
            n_line = 1
        csvout_writer.writerow(row)
        n_line += 1

    csvout.close()
    tsvin.close()


def create_linguistic_feature(features_list, doc):
    tokens = tokenizer.tokenize(doc)
    tagged_tokens = nltk.pos_tag(tokens)
    features = np.zeros(len(features_list))

    for i, token in enumerate(tokens):
        tag = tagged_tokens[i][1]
        for j, feature in enumerate(features_list):
            if token.lower() in feature or tag in feature:
                features[j] += 1
                break

    features = np.divide(features, len(tokens))

    return features
