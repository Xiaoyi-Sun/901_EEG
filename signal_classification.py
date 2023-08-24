# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:10
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : signal_classification.py
# @Software: PyCharm
import os
from idlelib import history
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc

def process_gsr_data(filename, ground_truth_file):
    # Read the CSV file
    df = pd.read_csv(filename)

    # Group the data by 'Respondent' and 'SourceStimuliName' (video stimulus)
    grouped = df.groupby(['Respondent', 'SourceStimuliName'])

    # Initialize an empty list to store segment DataFrames
    segmented_data = []

    # Define the number of segments per video
    segments_per_video = 55

    # Define the list of features to calculate
    feature_functions = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'skewness': skew,
        'kurtosis': kurtosis,
        'range': lambda x: np.max(x) - np.min(x),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'cv': lambda x: np.std(x) / np.mean(x),
        'percentile_25': lambda x: np.percentile(x, 25),
        'percentile_75': lambda x: np.percentile(x, 75),
        'pa_mean': lambda x: x.mean(),
        'peak_ct': lambda x: (x * 100).value_counts().max(),
        'peak_sum': lambda x: (x * 100).value_counts().max() * (x * 100).max()
    }

    # Iterate through each group (respondent and video stimulus)
    for group_key, group_df in grouped:
        respondent, stimulus = group_key

        # Calculate the number of segments for this video
        num_segments = len(group_df) // segments_per_video

        # Split the data into segments
        segments = [group_df.iloc[i:i + num_segments] for i in range(0, len(group_df), num_segments)]

        # Extract necessary features from each segment
        for segment_df in segments:
            segment_features = {
                'Respondent': respondent,
                'SourceStimuliName': stimulus
            }

            # Calculate statistical features for 'Phasic Signal' and 'Tonic Signal'
            for col in ['Phasic Signal', 'Tonic Signal']:
                for feature_name, feature_func in feature_functions.items():
                    feature_value = feature_func(segment_df[col])
                    segment_features[f'{col}_{feature_name}'] = feature_value

            # Append the segment features to the list
            segmented_data.append(segment_features)

    # Create a DataFrame from the list of segment features
    segmented_data_df = pd.DataFrame(segmented_data)

    # Copy the segmented data DataFrame
    copy = segmented_data_df

    # Read ground truth data
    ground_truth_df = pd.read_csv(ground_truth_file)

    # Merge the 'Arousal' column from ground truth based on 'SourceStimuliName' and 'Respondent'
    merged_df = copy.merge(ground_truth_df[['Participant', 'Stimulus_Name', 'Valence', ' Arousal']],
                           left_on=['Respondent', 'SourceStimuliName'],
                           right_on=['Participant', 'Stimulus_Name'])

    merged_df.drop(columns=['Participant', 'Stimulus_Name'], inplace=True)

    merged_df = merged_df.fillna(0)

    # Select features (exclude 'Respondent', 'SourceStimuliName', and ' Arousal')
    features = merged_df.drop(['Respondent', 'SourceStimuliName', ' Arousal', 'Valence'], axis=1)
    target = merged_df[' Arousal']

    return features, target, merged_df


def plot_learning_curve(model, X_train, y_train, X_test, y_test, save_filename=None):
    train_errors, test_errors = [], []

    # Define a range of training set sizes
    training_sizes = np.linspace(0.1, 1.0, 10)

    for size in training_sizes:
        # Select a subset of the training data
        subset_size = int(size * len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]

        # Fit the model to the subset
        model.fit(X_subset, y_subset)

        # Make predictions on both training and test data
        y_train_pred = model.predict(X_subset)
        y_test_pred = model.predict(X_test)

        # Calculate RMSE for the predictions
        train_rmse = np.sqrt(mean_squared_error(y_subset, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
    save_filename = "/Users/sunxiaoyi/PycharmProjects/901_EEG/figure_name.png"
    save_directory = os.path.dirname(save_filename)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, train_errors, label="Training RMSE")
    plt.plot(training_sizes, test_errors, label="Testing RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Learning Curve")
    plt.grid(True)
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')

    plt.show()


def train_evaluate_models(features, target):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    models = []

    # Define a range of models and their hyperparameter spaces (e.g., for Random Forest)
    param_grid_rf = {
        'n_estimators': [10, 50],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
    }

    # Random Forest Regressor
    grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_model_rf = grid_search_rf.best_estimator_
    models.append(("RandomForest", best_model_rf))

    y_pred_rf = best_model_rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"Random Forest MAE: {mae_rf:.2f}")
    print(f"Random Forest RMSE: {rmse_rf:.2f}")

    # Gradient Boosting Regressor
    param_grid_gb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    grid_search_gb = GridSearchCV(GradientBoostingRegressor(), param_grid_gb, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search_gb.fit(X_train, y_train)
    best_model_gb = grid_search_gb.best_estimator_
    models.append(("GradientBoosting", best_model_gb))

    y_pred_gb = best_model_gb.predict(X_test)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    print(f"Gradient Boosting MAE: {mae_gb:.2f}")
    print(f"Gradient Boosting RMSE: {rmse_gb:.2f}")

    # Bayesian Ridge Regressor
    bayesian_reg = BayesianRidge()
    param_grid_bayesian = {
        'n_iter': [100, 200],
        'tol': [1e-6, 1e-5],
    }
    grid_search_bayesian = GridSearchCV(bayesian_reg, param_grid_bayesian, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search_bayesian.fit(X_train, y_train)
    best_bayesian_reg = grid_search_bayesian.best_estimator_
    models.append(("BayesianRidge", best_bayesian_reg))

    y_pred_bayesian = best_bayesian_reg.predict(X_test)
    mae_bayesian = mean_absolute_error(y_test, y_pred_bayesian)
    rmse_bayesian = np.sqrt(mean_squared_error(y_test, y_pred_bayesian))
    print(f"Bayesian Ridge MAE: {mae_bayesian:.2f}")
    print(f"Bayesian Ridge RMSE: {rmse_bayesian:.2f}")

    # K-Nearest Neighbors Regressor
    param_grid_knn = {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    }
    grid_search_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search_knn.fit(X_train, y_train)
    best_model_knn = grid_search_knn.best_estimator_
    models.append(("KNeighbors", best_model_knn))

    y_pred_knn = best_model_knn.predict(X_test)
    mae_knn = mean_absolute_error(y_test, y_pred_knn)
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    print(f"K-Nearest Neighbors MAE: {mae_knn:.2f}")
    print(f"K-Nearest Neighbors RMSE: {rmse_knn:.2f}")

    # Decision Tree Regressor
    param_grid_tree = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    grid_search_tree = GridSearchCV(DecisionTreeRegressor(), param_grid_tree, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search_tree.fit(X_train, y_train)
    best_model_tree = grid_search_tree.best_estimator_
    models.append(("DecisionTree", best_model_tree))

    y_pred_tree = best_model_tree.predict(X_test)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
    print(f"Decision Tree MAE: {mae_tree:.2f}")
    print(f"Decision Tree RMSE: {rmse_tree:.2f}")

    # Create a neural network model using Keras
    def create_model(learning_rate=0.001, batch_size=32):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae', 'mse'])
        return model

    tuner = RandomizedSearchCV(
        tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model),
        param_distributions={
            'learning_rate': [0.001],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150],
        },
        scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1
    )

    tuner.fit(X_train, y_train)
    best_keras_model = tuner.best_estimator_

    models.append(("KerasNN", best_keras_model))

    y_pred_keras = best_keras_model.predict(X_test)
    mae_keras = mean_absolute_error(y_test, y_pred_keras)
    rmse_keras = np.sqrt(mean_squared_error(y_test, y_pred_keras))
    print(f"Keras Neural Network MAE: {mae_keras:.2f}")
    print(f"Keras Neural Network RMSE: {rmse_keras:.2f}")

    # Find the best model
    best_mae = float('inf')
    best_model = None
    best_y_pred = None
    best_model_name = None

    for model_name, model in models:
        if model_name == "KerasNN":
            mae = mae_keras
            y_pred = y_pred_keras
        elif model_name == "RandomForest":
            mae = mae_rf
            y_pred = y_pred_rf
        elif model_name == "GradientBoosting":
            mae = mae_gb
            y_pred = y_pred_gb
        elif model_name == "BayesianRidge":
            mae = mae_bayesian
            y_pred = y_pred_bayesian
        elif model_name == "KNeighbors":
            mae = mae_knn
            y_pred = y_pred_knn
        elif model_name == "DecisionTree":
            mae = mae_tree
            y_pred = y_pred_tree

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_y_pred = y_pred
            best_model_name = model_name

    print(f"Best Model: {best_model_name}")
    print(f"Best Model: {best_mae}")
    rmse = np.sqrt(mean_squared_error(y_test, best_y_pred))
    print(f"Best Model: {best_model_name}")
    print(f"Best Model RMSE: {rmse:.2f}")
    # Plot and save loss curve for the best model

    plot_learning_curve(best_model, X_train, y_train, X_test, y_test)

# Return the best model, its predicted target values, and the name of the best model
    return best_model, best_y_pred





