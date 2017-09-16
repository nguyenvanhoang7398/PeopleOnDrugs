import math, csv, os, pickle, random, nltk, string
import numpy as np
from nltk.tokenize import word_tokenize

from constants import affective_features_default_path, affective_word_dict_default_path, schema


def save_data(path, py_object):
    with open(path, 'wb') as f:
        pickle.dump(py_object, f)


# Load data from existing pickle
def load_data(pickle_in):
    with open(pickle_in, 'rb') as f:
        contents = pickle.load(f)
    return contents


def isfile(path):
    return os.path.isfile(path)


def divide_batches(fin, n_batches, size):
    print("Divide", fin, "into", n_batches, "batches.")
    tsvin = open(fin + ".tsv", mode='rt', encoding="ISO-8859-1")
    name = fin.split("/")[-1]
    name = name.split(".")[0]
    tsvin_writer = csv.reader(tsvin, delimiter='\t')
    dir = fin + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    line_per_file = math.ceil(float(size) / n_batches)
    n_out = 1
    n_line = 1
    csvout = open(dir + name + "_" + str(n_out) + ".tsv", mode="wt", encoding="ISO-8859-1")
    csvout_writer = csv.writer(csvout, delimiter='\t', lineterminator='\n')
    for row in tsvin_writer: 
        if n_line > line_per_file:
            n_out += 1
            csvout.close()
            csvout = open(dir + name + "_" + str(n_out) + ".tsv", mode="wt", encoding="ISO-8859-1")
            csvout_writer = csv.writer(csvout, delimiter='\t', lineterminator='\n')
            n_line = 1
        csvout_writer.writerow(row)
        n_line += 1

    csvout.close()
    tsvin.close()


def create_stylistic_feature_vector(features_list, doc):
    tokens = word_tokenize(doc)
    tagged_tokens = nltk.pos_tag(tokens)
    features = np.zeros(len(features_list))
    num_toks = 0

    for i, token in enumerate(tokens):
        tag = tagged_tokens[i][1]
        for j, feature in enumerate(features_list):
            if token.lower() in feature or tag in feature:
                features[j] += 1
                break
        if token not in string.punctuation: # increase number of tokens if token is not a punctuation
            num_toks += 1

    if num_toks != 0:
        features = np.divide(features, num_toks)

    return features


def create_affective_feature_vector(doc):
    if isfile(affective_word_dict_default_path): # check if affective word dict has been created
        affective_word_dict, inv_affective_word_dict, affective_category_list = load_data(affective_word_dict_default_path)
    else: # if not, create a new one
        affective_word_dict, inv_affective_word_dict, affective_category_list = create_affective_word_dict()
        print("Affected word dictionary loaded at " + affective_word_dict_default_path)
    tokens = word_tokenize(doc) # tokenize doc
    features = np.zeros(len(affective_category_list)) # create zeros feature vector of size affective categories
    num_toks = 0

    for i, token in enumerate(tokens):
        if token in affective_word_dict.keys():
            for category in affective_word_dict[token]: # iterate through all categories of a word
                features[affective_category_list.index(category)] += 1
        if token not in string.punctuation: # increase number of tokens if token is not a punctuation
            num_toks += 1
    if num_toks != 0:
        features = np.divide(features, num_toks)

    return features


def create_train_test_data(dataset, test_size=0.1):
    random.shuffle(dataset)
    dataset = np.array(dataset)
    testing_size = int(test_size*len(dataset))

    train_inputs = list(dataset[:,0][:-testing_size])
    train_labels = list(dataset[:,1][:-testing_size])
    test_inputs = list(dataset[:,0][-testing_size:])
    test_labels = list(dataset[:,1][-testing_size:])

    return train_inputs, train_labels, test_inputs, test_labels


def convert_tsv_csv(file_in, file_out):
    tsvin = open(file_in, mode='rt', encoding="ISO-8859-1")
    tsvin_reader = csv.reader(tsvin, delimiter='\t')
    csvout = open(file_out, mode="wt", encoding="ISO-8859-1")
    csvout_writer = csv.writer(csvout, delimiter=',', lineterminator='\n')
    name = file_in.split("/")[-1]
    i = 1
    for row in tsvin_reader:
        if i == 1:
            csvout_writer.writerow(schema[name])
            i += 1
        csvout_writer.writerow(row)
        i += 1

    csvout.close()
    tsvin.close()


def create_affective_word_dict(fin=affective_features_default_path, fout=affective_word_dict_default_path):
    csvin = open(fin, mode='rt', encoding="ISO-8859-1")
    csvin_reader = csv.reader(csvin, delimiter=',')
    affective_word_dict = {}
    inv_affective_word_dict = {}
    affective_category_list = []

    for i, row in enumerate(csvin_reader): # iterate through file
        if i == 0:
            continue
        affective_category = row[0] # parse affective category
        affective_word_list = row[1] # parse affective word list
        for tok in word_tokenize(affective_word_list): # tokenize
            if tok not in string.punctuation: # eliminate punctuations
                if tok not in affective_word_dict.keys(): # check if token's alr in dict
                    affective_word_dict[tok] = set() # create new set, we use set to avoid duplicates
                if affective_category not in inv_affective_word_dict.keys(): # do the same w/ categories
                    inv_affective_word_dict[affective_category] = set()
                    affective_category_list += [affective_category]

                affective_word_dict[tok].add(affective_category)
                inv_affective_word_dict[affective_category].add(tok)

    save_data(path=fout, py_object=(affective_word_dict, inv_affective_word_dict, affective_category_list)) # save both forward and inverse dicts
    print("Affective word dict saved at " + fout)
    return affective_word_dict, inv_affective_word_dict, affective_category_list
