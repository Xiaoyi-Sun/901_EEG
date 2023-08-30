import pandas as pd
from keras import Sequential

from signal_classification import process_gsr_data, train_evaluate_models
from data_preprocessing_merge import group_data_v_new, group_data_v_old, get_raw_file, modify_respondent, concatenate_dataframes, extrac_data_v_old, extract_data_v_new
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from keras.models import save_model







# Create a dictionary to store repositories
repository_data = {}

# Read repositories from the text file and store them in the dictionary
with open("repository_2.txt", "r") as txt_file:
    for line in txt_file:
        name, url = line.strip().split(',')
        repository_data[name] = url

# Access repositories using the dictionary
repository_name_1 = "data_folder_old"
data_folder_old = repository_data[repository_name_1]

repository_name_2 = "data_folder_new"
data_folder_new = repository_data[repository_name_2]

repository_name_3 = "output_data"
output_data = repository_data[repository_name_3]

repository_name_4 = "old_data"
old_data = repository_data[repository_name_4]

repository_name_5 = "new_data"
new_data = repository_data[repository_name_5]

repository_name_6 = "old_new_combined_data"
old_new_combined_data = repository_data[repository_name_6]

repository_name_7 = "ready_data"
ready_data = repository_data[repository_name_7]

repository_name_8 = "ground_truth"
ground_truth = repository_data[repository_name_8]

repository_name_9 = "predicted_data"
predicted_data = repository_data[repository_name_9]


while True:
    print("\nChoose a function:")
    print("1. select old version of data v1 only file and convert them to csv and merge them to one csv file")
    print("2. select new version of data v5 and vf file and convert them to csv and merge them to one csv file")
    print("3. merge new data to existing dataframe")
    print("4. one off button for data processing, import ground truth, run model comparison and get the predicted results")
    print("5. plot time series graph with predicted results")
    print("6. Exit")


    choice = input("Enter the number of the function you want to choose: ")

    if choice == "1":
        df = group_data_v_old(data_folder_old)
        df.to_csv(output_data, index=False)
    elif choice == "2":
        df = group_data_v_new(data_folder_new)
        df.to_csv(output_data, index=False)
    elif choice == "3":
        updated_data_df = concatenate_dataframes(old_data, new_data)
        updated_data_df.to_csv(old_new_combined_data, index=False)
    elif choice == "4":
        features, target, merged_df = process_gsr_data(ready_data, ground_truth)
        best_model, y_pred = train_evaluate_models(features, target)

        split_index = int(len(merged_df) * 0.8)

        # Split the DataFrame into the first 80% and the last 20% based on the index
        merged_df_first_80 = merged_df.iloc[:split_index]
        merged_df_last_20 = merged_df.iloc[split_index:]

        # Add the predicted arousal values to the last 20% of the DataFrame
        merged_df_last_20['predicted_arousal'] = y_pred
        merged_df_last_20.to_csv(predicted_data, index=False)
        save_dir = 'saved_info'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
                # Create a list of unique participants
        # Create a directory to save the plots
        save_dir = 'scatterplots'
        os.makedirs(save_dir, exist_ok=True)

        # Define marker style
        marker_style = 'o'  # You can customize the marker style here

        # Define colors for the two types of plots
        colors = ['blue', 'red']  # Different colors for Valence/Arousal and Valence/Predicted Arousal

        # Get unique participants and unique stimuli
        unique_participants = merged_df_last_20['Respondent'].unique()
        unique_stimuli = merged_df_last_20['SourceStimuliName'].unique()

        # Loop through each unique participant and stimulus
        for participant in unique_participants:
            for stimulus in unique_stimuli:
                # Create a figure and axis for the current participant and stimulus
                fig, ax = plt.subplots(figsize=(8, 6))

                # Select data for the current participant and stimulus
                data = merged_df_last_20[(merged_df_last_20['Respondent'] == participant) & (merged_df_last_20['SourceStimuliName'] == stimulus)]

                # Scatterplot 1: Valence vs Arousal (Blue)
                ax.scatter(data['Valence'], data[' Arousal'], c=colors[0], marker=marker_style, label='Valence/Arousal', alpha=0.5)

                # Scatterplot 2: Valence vs Predicted Arousal (Red)
                ax.scatter(data['Valence'], data['predicted_arousal'], c=colors[1], marker=marker_style, label='Valence/Predicted Arousal', alpha=0.5)

                # Label each point with the stimulus name
                for x, y, stimulus_name in zip(data['Valence'], data['predicted_arousal'], data['SourceStimuliName']):
                    ax.text(x, y, stimulus_name, fontsize=8)

                # Set axis labels
                ax.set_xlabel('Valence')
                ax.set_ylabel('Arousal')

                # Add a legend
                ax.legend()

                # Set the title for the current plot
                ax.set_title(f'Scatterplot for Participant {participant} - Stimulus {stimulus}')

                # Save the plot for the current participant and stimulus
                plt.savefig(os.path.join(save_dir, f'scatterplot_{participant}_stimulus_{stimulus}.png'))

                # Close the figure for the current participant and stimulus
                plt.close()

        if isinstance(best_model, Sequential):
            # It's a Keras (deep learning) model
            model_save_path = os.path.join(save_dir, 'best_deep_learning_model')
            best_model.save(model_save_path)  # Save the deep learning model
        else:
            # It's a machine learning model
            model_save_path = os.path.join(save_dir, 'best_machine_learning_model.joblib')
            dump(best_model, model_save_path)  # Save the machine learning model


    elif choice == "5":
        df = pd.read_csv(predicted_data)

        # Convert the 'Timestamp' column to a datetime object (if it's not already)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Get unique participants and unique stimuli
        unique_participants = df['Respondent'].unique()
        unique_stimuli = df['SourceStimuliName'].unique()

        # Create a directory to save the individual time series plots
        save_dir = 'time_series_plots'
        os.makedirs(save_dir, exist_ok=True)

        # Loop through each unique participant and stimulus
        for participant in unique_participants:
            for stimulus in unique_stimuli:
                # Create a figure and axis for the current participant and stimulus
                plt.figure(figsize=(12, 6))
                ax = plt.gca()

                # Select data for the current participant and stimulus
                data = df[(df['Respondent'] == participant) & (df['SourceStimuliName'] == stimulus)]

                # Plot 'Arousal' and 'predicted_arousal' over time for the current combination
                ax.plot(data['Timestamp'], data[' Arousal'], label='Arousal', linestyle='-', marker='o', markersize=4)
                ax.plot(data['Timestamp'], data['predicted_arousal'], label='Predicted Arousal', linestyle='-', marker='o', markersize=4)

                # Add labels and title
                plt.xlabel('Timestamp')
                plt.ylabel('Value')
                plt.title(f'Time Series Data for Participant {participant} - Stimulus {stimulus}')
                plt.grid(True)

                # Add a legend
                plt.legend()

                # Save the time series plot for the current participant and stimulus
                plot_filename = os.path.join(save_dir, f'time_series_plot_{participant}_stimulus_{stimulus}.png')
                plt.savefig(plot_filename)

                # Close the figure
                plt.close()
    elif choice == "6":
        exit
