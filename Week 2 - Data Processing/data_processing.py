import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_data(filename):
    df = pd.read_csv(filename, header=None)
    data = df.values
    data[:, 4] = np.where(data[:, 4] == 'Iris-setosa', 1, np.where(data[:, 4] == 'Iris-versicolor', 2, 3))
    return data

def get_mean_and_sd(data):
    mean_value = np.mean(data, axis=0)
    sd_value = np.std(data, axis=0)
    return mean_value, sd_value


def get_class_distribution(class_data):
    unique_classes = np.unique(class_data)
    placeholder = {}
    for _class in unique_classes:
        placeholder[int(_class)] = np.count_nonzero(class_data==_class)
    return placeholder


def generate_hist(feature_data, feature_name):
    fig = plt.figure()
    plt.hist(feature_data, bins=15)
    plt.title(feature_name)
    fig.savefig(f"{feature_name}.png", bbox_inches='tight')
    plt.close(fig)


def train_test_split(X, y, train_ratio, shuffle=True):
    data_size = len(X)
    split_index = int(train_ratio*data_size)
    
    if shuffle:    
        order = np.random.permutation(np.arange(0, data_size))
        x_train = X[order[:split_index]]
        y_train = y[order[:split_index]]
        x_test = X[order[split_index:]]
        y_test = y[order[split_index:]]
    else:
        x_train = X[:split_index]
        y_train = y[:split_index]
        x_test = X[split_index:]
        y_test = y[split_index:]
    
    return x_train, y_train, x_test, y_test



if __name__ == '__main__':

    # Q1 -- Read Data
    filename = 'data/iris.data'
    iris_data = import_data(filename)
    iris_x = iris_data[:, :4].astype('float') 
    iris_y = iris_data[:, 4].astype('float')

    # Q2 -- Feature Mean and Std
    feature_mean, feature_sd = get_mean_and_sd(iris_x)
    print(f"Feature mean values: {feature_mean}")
    print(f"Feature sd values: {feature_sd}")

    # Q3 -- Class Distribution
    class_dist = get_class_distribution(iris_y)
    print(f"The class distribution of labels: {class_dist}" )

    # Q4 -- Plot Histograms
    for i in range(iris_x.shape[1]):
        feature_name = f'feature_{i+1}'
        generate_hist(iris_x[:, i], feature_name=feature_name)


    # Q5 -- Train & Test split
    x_train_random, y_train_random, x_test_random, y_test_random = train_test_split(iris_x, iris_y, 
                                                                        train_ratio=0.6,
                                                                        shuffle=True)

    print(x_train_random.shape, y_train_random.shape, x_test_random.shape, y_test_random.shape)
    print(y_test_random)

    x_train, y_train, x_test, y_test = train_test_split(iris_x, iris_y, 
                                                        train_ratio=0.6,
                                                        shuffle=False)

    print(y_test)

    

