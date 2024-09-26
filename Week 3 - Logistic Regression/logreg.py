import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix


## Functions
def read_data_from_csv(filename, test_ratio=0.4, shuffle=True):

    # Read as DataFrame
    df = pd.read_csv(filename, header=None)
    data = df.values

    # Extract Input and Target Arrays
    data_x = data[:, :8]
    data_y = data[:, 8]

    # TODO: Scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    x_scaled = scaler.transform(data_x)

    # The above two steps can also be combined into one step
    # x_scaled = scaler.fit_transform(data_x)

    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, data_y, test_size=test_ratio, shuffle=shuffle)

    # return data_x, data_y
    return x_train, x_test, y_train, y_test


def generate_roc_curve(y_true, y_pred, filename='roc_curve.png'):
    
    # TODO: Get true Positive and false positive rates
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot the roc curve for the model
    fig, ax = plt.subplots()
    ax.plot(tpr, fpr, label='logistic-regression')
    ax.set_title('ROC Curve')
    ax.set_ylabel('False Positive Rate')
    ax.set_xlabel('True Positive Rate')
    plt.legend()

    # Save figure
    fig.savefig(filename, bbox_inches='tight')



def evaluate_model(model, x, y, dataset='train'):

    # TODO: Generate predictions
    y_pred = model.predict(x)

    # Accuracy score
    acc = accuracy_score(y, y_pred)

    # f1_score
    f1 = f1_score(y, y_pred)

    # ROC AUC curve
    auc = roc_auc_score(y, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)

    # TODO: Generate ROC Curve
    y_probs = model.predict_proba(x)[:, 0]
    generate_roc_curve(y, y_probs, filename=f"{dataset}_roc.png")

    return acc, f1, auc, cm


def train_logistic_regression(x_train, y_train, n_iter=150, solver='saga', penalty=None):

    # TODO: Define the model
    logreg_model = LogisticRegression(max_iter=n_iter, penalty=penalty, solver=solver)
   
    # TODO: Train the model
    logreg_model.fit(x_train, y_train)

    # TODO: Predict
    y_pred = logreg_model.predict(x_train)

    # TODO: Compute accuracy
    acc = accuracy_score(y_train, y_pred)

    print(f"Model accuracy -- train: {acc:.2f}")

    return logreg_model



if __name__ == '__main__':

    # Read data from CSV file
    filename = 'data/pima-indians-diabetes.csv'
    x_train, x_test, y_train, y_test = read_data_from_csv(filename, test_ratio=0.3)
    # print(x_test.mean(axis=0), x_test.std(axis=0))

    # Train Logistic Regression
    logreg_model = train_logistic_regression(x_train, y_train, n_iter=250, penalty=None)

    # Evaluate model
    train_acc, train_f1, train_auc, train_cm = evaluate_model(logreg_model, x_train, y_train, dataset='train')
    test_acc, test_f1, test_auc, test_cm = evaluate_model(logreg_model, x_test, y_test, dataset='test')

    print(f"-- Train set performance -- \nacc: {train_acc:.2f}, f1: {train_f1:.2f}, auc: {train_auc:.2f}")
    print("Confusion matrix: ")
    print(train_cm)
    
    print(f"-- Test set performance -- \nacc: {test_acc:.2f}, f1: {test_f1:.2f}, auc: {test_auc:.2f}")
    print("Confusion matrix: ")
    print(test_cm)

    # TODO: Cross validation
    # Use cross_validate function from sklearn to perform cross validation
    


