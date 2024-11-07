import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor, 
                              AdaBoostRegressor)

from xgboost import XGBRegressor
pd.set_option("display.precision", 3)



# Functions
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

    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, data_y, test_size=test_ratio, shuffle=shuffle)

    # return data_x, data_y
    return x_train, x_test, y_train, y_test


def train_stacking_model(x_train, y_train, model='decision_tree'):

    # Initialize the base models
    base_models = [
        DecisionTreeRegressor(max_depth=3),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(n_estimators=100)
    ]

    # Initialize the meta model
    meta_model = DecisionTreeRegressor(max_depth=3)

    # Train the base models
    base_model_predictions = []
    for base_model in base_models:
        base_model.fit(x_train, y_train)
        base_model_predictions.append(base_model.predict(x_train))

    # Stack the predictions
    stacked_predictions = np.column_stack(base_model_predictions)

    # Train the meta model
    meta_model.fit(stacked_predictions, y_train)

    return base_models, meta_model


def generate_stacking_model_predictions(base_models, meta_model, x_test, y_test):

    # Generate predictions from base models
    base_model_predictions = []
    for base_model in base_models:
        base_model_predictions.append(base_model.predict(x_test))

    # Stack the predictions
    stacked_predictions = np.column_stack(base_model_predictions)

    # Generate predictions from meta model
    y_pred = meta_model.predict(stacked_predictions)

    return y_pred


def train_regression_model(x_train, y_train, model='random_forest'):

    # Initialize the model
    if model == 'random_forest':
        ensemble_model = RandomForestRegressor(n_estimators=20, max_depth=3)
    elif model == 'gradient_boosting':
        ensemble_model = GradientBoostingRegressor(n_estimators=20)
    elif model == 'adaboost':
        ensemble_model = AdaBoostRegressor(n_estimators=20)
    elif model == 'xgboost':
        ensemble_model = XGBRegressor(n_estimators=20)
    elif model == 'decision_tree':
        ensemble_model = DecisionTreeRegressor(max_depth=3)
    else:
        raise ValueError(f"Invalid model: {model}")

    # Train the model
    ensemble_model.fit(x_train, y_train)

    return ensemble_model


def generate_model_predictions(ensemble_model, x_test):

    # Generate predictions
    y_pred = ensemble_model.predict(x_test)

    return y_pred

def evaluate_predictions(y_true, y_pred):

    # Calculate the Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate the R2 Score
    r2 = 1 - mse / np.var(y_true)

    return mse, r2



if __name__ == '__main__':

    # Read the data
    filename = 'data/ENB2012_data.csv'
    x_train, x_test, y_train, y_test = read_data_from_csv(filename, test_ratio=0.4)

    # Train the models
    ensemble_models = ['decision_tree', 'random_forest', 'adaboost', 'xgboost']
    trained_models = {}

    # Initialize the results data
    train_res_data = {}
    test_res_data = {}
    n_exp = 10

    for model_name in ensemble_models:

        print("Training model: ", model_name)

        # Train Decision Tree Model
        train_mse = np.zeros(n_exp)
        train_r2 = np.zeros(n_exp)

        test_mse = np.zeros(n_exp)
        test_r2 = np.zeros(n_exp)

        for n in range(n_exp):

            trained_models[model_name] = train_regression_model(x_train, y_train, model=model_name)
            y_pred = generate_model_predictions(trained_models[model_name], x_train)
            train_mse[n], train_r2[n] = evaluate_predictions(y_train, y_pred)

            y_pred = generate_model_predictions(trained_models[model_name], x_test)
            test_mse[n], test_r2[n] = evaluate_predictions(y_test, y_pred)

        train_res_data[model_name] = {'mse_mean': train_mse.mean(),
                                'mse_std': train_mse.std(),
                                'r2_mean': train_r2.mean(),
                                'r2_std': train_r2.std()}
        
        test_res_data[model_name] = {'mse_mean': test_mse.mean(),
                                'mse_std': test_mse.std(),
                                'r2_mean': test_r2.mean(),
                                'r2_std': test_r2.std()}

    
    # Train Stacking Model
    train_mse = np.zeros(n_exp)
    train_r2 = np.zeros(n_exp)

    test_mse = np.zeros(n_exp)
    test_r2 = np.zeros(n_exp)

    for n in range(n_exp):
        base_models, meta_model = train_stacking_model(x_train, y_train)
        y_pred = generate_stacking_model_predictions(base_models, meta_model, x_train, y_train)
        train_mse[n], train_r2[n] = evaluate_predictions(y_train, y_pred)

        y_pred = generate_stacking_model_predictions(base_models, meta_model, x_test, y_test)
        test_mse[n], test_r2[n] = evaluate_predictions(y_test, y_pred)
    
    train_res_data['stacking'] = {'mse_mean': train_mse.mean(),
                                'mse_std': train_mse.std(),
                                'r2_mean': train_r2.mean(),
                                'r2_std': train_r2.std()}

    test_res_data['stacking'] = {'mse_mean': test_mse.mean(),
                                'mse_std': test_mse.std(),
                                'r2_mean': test_r2.mean(),
                                'r2_std': test_r2.std()}


    # Display the results
    res_df = pd.DataFrame(train_res_data)
    print("\nTrain data performance results: ")
    print(res_df)

    res_df = pd.DataFrame(test_res_data)
    print("\nTest data performance results: ")
    print(res_df)

    # Compare feature importances XGBoost
    xgb_importances = trained_models['xgboost'].feature_importances_
    xgb_importances = xgb_importances / np.sum(xgb_importances)

    tree_importances = trained_models['decision_tree'].feature_importances_
    tree_importances = tree_importances / np.sum(tree_importances)

    forrest_importances = trained_models['random_forest'].feature_importances_
    forrest_importances = forrest_importances / np.sum(forrest_importances)

    # Plot the feature importances
    # Number of features
    n_features = x_train.shape[1]

    # Create an array with the positions of the bars``
    indices = np.arange(n_features)

    # Width of a bar
    width = 0.25

    # Set seaborn color palette
    sns.set_palette("colorblind")

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")

    # Plot bars for Decision Tree
    plt.bar(indices, tree_importances, width, label='Decision Tree')

    # Plot bars for Random Forest, shifted by the width of a bar
    plt.bar(indices + width, forrest_importances, width, label='Random Forest')

    # Plot bars for XGBoost, shifted by twice the width of a bar
    plt.bar(indices + 2 * width, xgb_importances, width, label='XGBoost')

    # Add labels, legend, and save the plot
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.xticks(indices + width, range(n_features))
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()