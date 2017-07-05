from utils import create_linguistic_feature, save_data, load_data
import os, csv
from utils import feature_list
import numpy as np


def build_review_dataset(fin):
    csvin = open(fin, mode='rt')
    csvin_reader = csv.reader(csvin, delimiter='\t')
    dataset = []

    for i, row in enumerate(csvin_reader):
        print(i)
        author_id = row[0]
        review_id = row[1]
        print(review_id)
        review = row[2]
        feature_vector = create_linguistic_feature(features_list=feature_list, doc=review)
        dataset.append([author_id, review_id, feature_vector])

    fout = fin.split(".")[0] + "_dataset.pickle"
    save_data(path=fout, py_object=dataset)
    csvin.close()


def build_user_dataset(review_dataset_path, users_details_path, out_path, samples_per_files=3000):
    user_details_files = [f for f in os.listdir(users_details_path) if f.endswith('.csv')]
    review_dataset_files = [f for f in os.listdir(users_details_path) if f.endswith('.pickle')]
    review_dataset = []
    user_features_dict = {}
    for file in review_dataset_files:
        review_dataset.append(load_data(file))

    for review in review_dataset:
        user_id = review[0]
        review_vector = review[2]
        if user_id not in user_features_dict.keys():
            user_features_dict[user_id] = review_vector
        else:
            user_features_dict[user_id] = np.add(user_features_dict[user_id], review_vector)
    user_dataset = []

    for fin in user_details_files:
        csvin = open(fin, mode='rt')
        csvin_reader = csv.reader(csvin, delimiter='\t')

        for row in csvin_reader:
            user_id = row[0]
            num_reviews = row[3]
            num_likes = row[7]
            normalized_user_vector = np.divide(user_features_dict[user_id], num_reviews)
            normalized_likes = num_likes / num_reviews
            user_dataset.append([normalized_user_vector, normalized_likes])
        csvin.close()
    small_batch = []
    j = 1
    for i, data in enumerate(user_dataset):
        small_batch.append(data)
        if i % samples_per_files == 0:
            save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)
            j += 1
            small_batch = []
    save_data(out_path + "user_dataset_" + str(j) + ".pickle", small_batch)