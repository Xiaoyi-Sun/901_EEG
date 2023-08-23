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
with open("repository.txt", "r") as txt_file:
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
    print("5. Exit")


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
        save_dir = 'saved_info'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
                # Create a list of unique participants
        import matplotlib.pyplot as plt

        # Create a directory to save the plots
        save_dir = 'scatterplots'
        os.makedirs(save_dir, exist_ok=True)

        # Define marker style
        marker_style = 'o'  # You can customize the marker style here

        # Define colors for the two types of plots
        colors = ['blue', 'red']  # Different colors for Valence/Arousal and Valence/Predicted Arousal

        # Get unique participants
        unique_participants = merged_df_last_20['Respondent'].unique()

        # Loop through each unique participant
        for participant in unique_participants:
            # Create a figure and axis for the current participant
            fig, ax = plt.subplots(figsize=(8, 6))

            # Select data for the current participant
            participant_data = merged_df_last_20[merged_df_last_20['Respondent'] == participant]

            # Scatterplot 1: Valence vs Arousal (Blue)
            ax.scatter(participant_data['Valence'], participant_data[' Arousal'], c=colors[0], marker=marker_style, label='Valence/Arousal', alpha=0.5)

            # Scatterplot 2: Valence vs Predicted Arousal (Red)
            ax.scatter(participant_data['Valence'], participant_data['predicted_arousal'], c=colors[1], marker=marker_style, label='Valence/Predicted Arousal', alpha=0.5)

            # Label each point with the stimulus name
            for x, y, stimulus in zip(participant_data['Valence'], participant_data['predicted_arousal'], participant_data['SourceStimuliName']):
                ax.text(x, y, stimulus, fontsize=8)

            # Set axis labels
            ax.set_xlabel('Valence')
            ax.set_ylabel('Arousal')

            # Add a legend
            ax.legend()

            # Set the title for the current plot
            ax.set_title(f'Scatterplot for Participant {participant}')

            # Save the plot for the current participant
            plt.savefig(os.path.join(save_dir, f'scatterplot_{participant}.png'))

            # Close the figure for the current participant
            plt.close()



        if isinstance(best_model, Sequential):
            # It's a Keras (deep learning) model
            model_save_path = os.path.join(save_dir, 'best_deep_learning_model')
            best_model.save(model_save_path)  # Save the deep learning model
        else:
            # It's a machine learning model
            model_save_path = os.path.join(save_dir, 'best_machine_learning_model.joblib')
            dump(best_model, model_save_path)  # Save the machine learning model



        merged_df_last_20.to_csv(predicted_data, index=False)

# Alternatively, you can save the plot as an image file if needed, e.g., plt.savefig('participant_plot.png')
    elif choice == "5":
        exit
