import os
import pandas as pd
import numpy as np

np.random.seed(42)

PARENT_DIRNAME = os.path.expanduser("~/PRML-MidTerm-Project/")
df_train = pd.read_csv(PARENT_DIRNAME + "data/df_train.csv")
folder_path = PARENT_DIRNAME + "model_implementation/machine_learning_models/output/"

len_features = len(df_train.columns) - 1 # excluding the target column

binary_f1_scores = {'binary-focused': [], 'binary-unfocused': [], 'binary-drowsy': []}
multi_class_f1_scores = []

# Function to extract F1_Score and method from a result file
def extract_f1_score(file_path: str) -> pd.DataFrame:
    """
    Extract the F1_Score and method from the result file.

    Parameters:
    file_path (str): The path to the result file.

    Returns:
    pd.DataFrame: The DataFrame containing the 'Method', 'Model', and 'F1_Score' columns.
    """
    df = pd.read_csv(file_path)
    
    if 'Method' not in df.columns or 'F1_Score' not in df.columns:
        raise ValueError(f"File {file_path} is missing required columns ('Method' or 'F1_Score').")
    
    return df[['Method', 'Model', 'F1_Score', 'Details']]

# Loop through all the files and directories in the folder to find the best model
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Only process directories, not files
    if os.path.isdir(file_path):
        model_name = file_name.split('_')[:-1]
        model_name = '_'.join(model_name)

        for result in os.listdir(file_path):
            if result.split('.')[-1] != 'csv':
                continue
            
            file_csv = os.path.join(file_path, result)

            if 'binary' in result:
                df = extract_f1_score(file_csv)
                
                for idx, row in df.iterrows():
                    method = row['Method']
                    label = row['Model']
                    f1_score = row['F1_Score']
                    
                    # Set the details for PCA
                    if method == 'PCA':
                        details = f"{int(round(0.7 + 0.05 * (idx % 6)) * len_features)} components"
                    else:
                        details = row['Details']
                    
                    # Check the label and add it to the correct list
                    if label == 'binary-focused':
                        binary_f1_scores['binary-focused'].append((model_name, method, label, f1_score, details))
                    elif label == 'binary-unfocused':
                        binary_f1_scores['binary-unfocused'].append((model_name, method, label, f1_score, details))
                    elif label == 'binary-drowsy':
                        binary_f1_scores['binary-drowsy'].append((model_name, method, label, f1_score, details))
            
            elif 'multiclass' in result:
                df = extract_f1_score(file_csv)
                for idx, row in df.iterrows():
                    method = row['Method']
                    label = row['Model']
                    f1_score = row['F1_Score']
                    # Set the details for PCA
                    if method == 'PCA':
                        details = f"{int(round(0.7 + 0.05 * (idx % 6)) * len_features)} components"
                    else:
                        details = row['Details']
                    multi_class_f1_scores.append((model_name, method, label, f1_score, details))

# 1. Find the best model for each binary method
# Initialize the dictionary for binary models
best_results_for_each_model_bi = {
    'binary-focused': {},
    'binary-unfocused': {},
    'binary-drowsy': {}
}

# Check if the binary model keys exist and find the best model for each binary method
for label in binary_f1_scores:
    if binary_f1_scores[label]:
        best_results_for_each_model_bi[label] = max(binary_f1_scores[label], key=lambda x: x[-2])
    else:
        best_results_for_each_model_bi[label] = None

# Find the best model for multi-class classification
best_multi_class_model = max(multi_class_f1_scores, key=lambda x: x[-2], default=None)

# Prepare the data to export into a CSV file
results = {
    'Label Type': ['Binary (Focused)', 'Binary (Unfocused)', 'Binary (Drowsy)', 'Multi-Class'],
    'Method': [
        best_results_for_each_model_bi['binary-focused'][1] if best_results_for_each_model_bi['binary-focused'] else 'N/A',
        best_results_for_each_model_bi['binary-unfocused'][1] if best_results_for_each_model_bi['binary-unfocused'] else 'N/A',
        best_results_for_each_model_bi['binary-drowsy'][1] if best_results_for_each_model_bi['binary-drowsy'] else 'N/A',
        best_multi_class_model[1] if best_multi_class_model else 'N/A'
    ],
    'Best Model': [
        best_results_for_each_model_bi['binary-focused'][0] if best_results_for_each_model_bi['binary-focused'] else 'N/A',
        best_results_for_each_model_bi['binary-unfocused'][0] if best_results_for_each_model_bi['binary-unfocused'] else 'N/A',
        best_results_for_each_model_bi['binary-drowsy'][0] if best_results_for_each_model_bi['binary-drowsy'] else 'N/A',
        best_multi_class_model[0] if best_multi_class_model else 'N/A'
    ],
    'F1_Score': [
        best_results_for_each_model_bi['binary-focused'][-2] if best_results_for_each_model_bi['binary-focused'] else 'N/A',
        best_results_for_each_model_bi['binary-unfocused'][-2] if best_results_for_each_model_bi['binary-unfocused'] else 'N/A',
        best_results_for_each_model_bi['binary-drowsy'][-2] if best_results_for_each_model_bi['binary-drowsy'] else 'N/A',
        best_multi_class_model[-2] if best_multi_class_model else 'N/A'
    ],
    'Details': [
        best_results_for_each_model_bi['binary-focused'][-1] if best_results_for_each_model_bi['binary-focused'] else 'N/A',
        best_results_for_each_model_bi['binary-unfocused'][-1] if best_results_for_each_model_bi['binary-unfocused'] else 'N/A',
        best_results_for_each_model_bi['binary-drowsy'][-1] if best_results_for_each_model_bi['binary-drowsy'] else 'N/A',
        best_multi_class_model[-1] if best_multi_class_model else 'N/A'
    ]
}

# 2. Find the best result for each model based on the F1_Score and return 'Model', 'Method', 'F1_Score', and 'Details'
best_results_for_each_model_bi = {
    'binary-focused': {},
    'binary-unfocused': {},
    'binary-drowsy': {}
}

best_results_for_each_model_mc = {}

# For each binary label (focused, unfocused, drowsy), find the best method per model
for label, model_info in binary_f1_scores.items():
    for idx, (model_name, method, _, f1_score, details) in enumerate(model_info):
        # Apply PCA condition for details
        if method == 'PCA':
            details = f"{int(round(0.7 + 0.05 * (idx % 6)) * len_features)} components"
        if model_name not in best_results_for_each_model_bi[label] or f1_score > best_results_for_each_model_bi[label][model_name][-2]:
            best_results_for_each_model_bi[label][model_name] = (method, f1_score, details)

# For multi-class, find the best method per model
for idx, (model_name, method, _, f1_score, details) in enumerate(multi_class_f1_scores):
    # Apply PCA condition for details in multi-class case
    if method == 'PCA':
        details = f"{int(round(0.7 + 0.05 * (idx % 6)) * len_features)} components"
    if model_name not in best_results_for_each_model_mc or f1_score > best_results_for_each_model_mc[model_name][-2]:
        best_results_for_each_model_mc[model_name] = (method, f1_score, details)

# Prepare the data to export into a CSV file
results_for_each_model = []

# Add binary classification models
for label in ['binary-focused', 'binary-unfocused', 'binary-drowsy']:
    for model_name, (method, f1_score, details) in best_results_for_each_model_bi[label].items():
        results_for_each_model.append({
            'Model': model_name,
            'Label Type': label,
            'Best Method': method,
            'F1_Score': f1_score,
            'Details': details
        })

# Add multi-class models
for model_name, (method, f1_score, details) in best_results_for_each_model_mc.items():
    results_for_each_model.append({
        'Model': model_name,
        'Label Type': 'Multi-Class',
        'Best Method': method,
        'F1_Score': f1_score,
        'Details': details
    })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)
best_results_df = pd.DataFrame(results_for_each_model)

# Export the results to a CSV file
results_df.to_csv(folder_path + 'best_model_f1_scores.csv', index=False)
print("Best model F1 scores and details have been saved to 'best_model_f1_scores.csv'.")

best_results_df.to_csv(folder_path + 'best_results_for_each_model.csv', index=False)
print("Best results for each model have been saved to 'best_results_for_each_model.csv'.")