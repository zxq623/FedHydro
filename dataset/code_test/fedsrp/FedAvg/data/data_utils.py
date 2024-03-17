import numpy as np
from sklearn.model_selection import train_test_split


def split_data(X, y, train_size = 1.0):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[]}

    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)

        X_train, y_train = X[i], y[i]

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))

    print("Total number of samples:", sum(num_samples['train']))
    print("The number of train samples:", num_samples['train'])

    del X, y

    return train_data


def save_file(train_path, train_data):

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)

    print("Finish generating dataset.\n")
