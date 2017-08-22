from utils import create_linguistic_feature, save_data, load_data
import os, csv
from utils import feature_list
import numpy as np


def _build_review_dataset(fin):
    csvin = open(fin, mode='rt', encoding="ISO-8859-1")
    csvin_reader = csv.reader(csvin, delimiter='\t')
    dataset = []

    for i, row in enumerate(csvin_reader):
        author_id = row[0]
        review_id = row[1] 
        review = row[2]
        feature_vector = create_linguistic_feature(features_list=feature_list, doc=review)
        dataset.append([author_id, review_id, feature_vector])

    fout = fin.split(".")[0] + "_dataset.pickle"
    save_data(path=fout, py_object=dataset)
    csvin.close()


def build_review_dataset(review_path):
    review_files = [f for f in os.listdir(review_path) if f.endswith('.tsv')]
    for i, fin in enumerate(review_files):
        _build_review_dataset(review_path + fin)
        print("Finished creating", i, "/", len(review_files), "datasets")


def build_user_dataset(review_dataset_path, users_details_path, out_path, samples_per_files=3000):
    user_details_files = [f for f in os.listdir(users_details_path) if f.endswith('.tsv')]
    review_dataset_files = [f for f in os.listdir(review_dataset_path) if f.endswith('.pickle')]
    review_dataset = []
    user_features_dict = {}
    for i, file in enumerate(review_dataset_files):
        if (i+1) % 20:
            print("Loaded", str(i+1), "/", str(len(review_dataset_files)))
        dataset = load_data(review_dataset_path + file)
        review_dataset += dataset

    for review in review_dataset:
        user_id = review[0]
        review_vector = review[2]
        if user_id not in user_features_dict.keys():
            if not np.isnan(review_vector).any():
                user_features_dict[user_id] = review_vector
        elif not np.isnan(review_vector).any():
            user_features_dict[user_id] = np.add(user_features_dict[user_id], review_vector)
    user_dataset = []
    
    i = 0
    for fin in user_details_files:
        csvin = open(users_details_path + fin, mode='rt', encoding="ISO-8859-1")
        csvin_reader = csv.reader(csvin, delimiter='\t')

        for row in csvin_reader:
            if i % 500 == 0:
                print("Proceeded", i, "/ 15000 users")
            i += 1
            user_id = row[0]
            if user_id not in user_features_dict.keys():
                continue    
            num_reviews = int(row[3])
            num_likes = (row[7])
            if num_likes == 'null':
                num_likes = 0
            else:
                num_likes = int(row[7])
            normalized_user_vector = np.divide(user_features_dict[user_id], num_reviews)
            normalized_likes = num_likes / num_reviews   
            if not (np.isnan(normalized_user_vector).any() or np.isnan(normalized_likes)):
                user_dataset.append([normalized_user_vector, normalized_likes])
        csvin.close()
    small_batch = []
    j = 1
    for i, data in enumerate(user_dataset):
        small_batch.append(data)
        if (i+1) % samples_per_files == 0:
            save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)
            j += 1
            small_batch = []
    save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)
