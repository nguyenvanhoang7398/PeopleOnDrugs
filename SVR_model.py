import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression as LR
from utils import load_data, create_train_test_data
from sklearn.metrics import mean_squared_error
import os, math

user_dataset_path = "data/user_dataset/"

def load_user_dataset(user_dataset_path):
    user_dataset_files = [f for f in os.listdir(user_dataset_path) if f.endswith('.pickle')]
    dataset = []
    for fin in user_dataset_files:
        dataset += load_data(user_dataset_path + fin)

    return dataset

dataset = load_user_dataset(user_dataset_path)
train_inputs, train_labels, test_inputs, test_labels = create_train_test_data(dataset)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(train_inputs, train_labels)
pred_labels_svr_rbf = svr_rbf.predict(test_inputs)
rmse_svr_rbf = math.sqrt(mean_squared_error(test_labels, pred_labels_svr_rbf))
print("RMSE for SVR with RBF kernel:", rmse_svr_rbf)
print("R^2 for SVR:", svr_rbf.score(test_inputs, test_labels))

lr = LR()
lr.fit(train_inputs, train_labels)
pred_labels_lr = lr.predict(test_inputs)
rmse_lr = math.sqrt(mean_squared_error(test_labels, pred_labels_lr))
print("RMSE for LR:", rmse_lr)
print("R^2 for LR:", lr.score(test_inputs, test_labels))
  
