import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

    # Scale the data
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
    
    # Get true Positive and false positive rates
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot the roc curve for the model
    fig, ax = plt.subplots()
    ax.plot(tpr, fpr, label='mlp-classifier')
    ax.set_title('ROC Curve')
    ax.set_ylabel('False Positive Rate')
    ax.set_xlabel('True Positive Rate')
    plt.legend()

    # save figure
    fig.savefig(filename, bbox_inches='tight')



def evaluate_model(model, x, y, run_name='train'):

    # Generate predictions
    y_pred = model.predict(x)

    # Accuracy score
    acc = accuracy_score(y, y_pred)

    # f1_score
    f1 = f1_score(y, y_pred)

    # ROC AUC curve
    auc = roc_auc_score(y, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)

    # Generate ROC Curve
    y_probs = model.predict_proba(x)[:, 0]
    generate_roc_curve(y, y_probs, filename=f"{run_name}_roc.png")

    return acc, f1, auc, cm



def train_mlp_classifier(x_train, y_train, hidden=10, n_iter=150, solver='adam', alpha=1e-4, learning_rate=1e-2):

    # TODO: Define the model
    mlp = MLPClassifier(hidden_layer_sizes=(hidden,),
                        max_iter=n_iter,
                        solver=solver,
                        alpha=alpha,
                        learning_rate_init=learning_rate,
                        random_state=123)
   
    # Train the model
    mlp.fit(x_train, y_train)

    # Predict
    y_pred = mlp.predict(x_train)

    # Compute accuracy
    acc = accuracy_score(y_train, y_pred)

    print(f"Model accuracy -- train: {acc:.2f}")

    return mlp



if __name__ == '__main__':

    # Read data from CSV file
    filename = 'data/pima-indians-diabetes.csv'
    x_train, x_test, y_train, y_test = read_data_from_csv(filename, test_ratio=0.4)
    # print(x_test.mean(axis=0), x_test.std(axis=0))



    # EVALUATE EFFECT OF HIDDEN UNITS ON MODEL PERFORMANCE
    hidden_performance = {}
    lr = 0.01
    n_iter = 500

    for h, hidden in enumerate(range(2, 100, 10)):

        print(f"\n\n Run :{h+1} -- Running for #hidden as: {hidden}")

        # Train MLP Classifier
        model = train_mlp_classifier(x_train, y_train, hidden=hidden, n_iter=n_iter, alpha=1e-2, learning_rate=lr)

        # Evaluate model
        train_acc, train_f1, train_auc, train_cm = evaluate_model(model, x_train, y_train, run_name=f'hidden_exp/train_hidden_{hidden}')
        test_acc, test_f1, test_auc, test_cm = evaluate_model(model, x_test, y_test, run_name=f'hidden_exp/test_hidden_{hidden}')

        print(f"-- Train set performance -- \nacc: {train_acc:.2f}, f1: {train_f1:.2f}, auc: {train_auc:.2f}")
        print(f"-- Test set performance -- \nacc: {test_acc:.2f}, f1: {test_f1:.2f}, auc: {test_auc:.2f}")

        hidden_performance[hidden] = {'train_f1': np.round(train_f1, 2), 'test_f1': np.round(test_f1, 2)}
    
    print("\nHidden units performance summary: ")
    
    # convert dictionary to DataFrame
    res = pd.DataFrame(hidden_performance)
    res.index.name = 'hidden_units'
    pprint(res)


    # EVALUATE EFFECT OF LEARNING RATE ON MODEL PERFORMANCE
    lr_performance = {}
    hidden = 6 
    for l, lr in enumerate(np.linspace(0.1, 1, 10).round(2)):

        print(f"\n\n Run: {l+1} -- Running for learning rate as: {lr}")

        # Train Logistic Regression
        model = train_mlp_classifier(x_train, y_train, hidden=hidden, n_iter=n_iter, alpha=1e-2, learning_rate=lr)

        # Evaluate model
        train_acc, train_f1, train_auc, train_cm = evaluate_model(model, x_train, y_train, run_name=f'lr_exp/train_lr_{lr}')
        test_acc, test_f1, test_auc, test_cm = evaluate_model(model, x_test, y_test, run_name=f'lr_exp/test_lr_{lr}')

        print(f"-- Train set performance -- \nacc: {train_acc:.2f}, f1: {train_f1:.2f}, auc: {train_auc:.2f}")
        # print("Confusion matrix: ")
        # print(train_cm)
        
        print(f"-- Test set performance -- \nacc: {test_acc:.2f}, f1: {test_f1:.2f}, auc: {test_auc:.2f}")
        # print("Confusion matrix: ")
        # print(test_cm)

        lr_performance[lr] = {'train_f1': np.round(train_f1, 2), 'test_f1': np.round(test_f1, 2)}
    
    print("\nLearning rate performance summary: ")
    res = pd.DataFrame(lr_performance)
    res.index.name = 'learning_rate'
    pprint(res)

