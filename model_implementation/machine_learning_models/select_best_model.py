import os
import pandas as pd

PARENT_DIRNAME = os.path.expanduser("~/PRML-MidTerm-Project/")
folder_path = PARENT_DIRNAME + "model_implementation/machine_learning_models/output/"

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

# Loop through all the files and directories in the folder
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
                        details = round(0.7 + 0.05 * divmod(idx, 6)[0], 2)
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
                for _, row in df.iterrows():
                    method = row['Method']
                    label = row['Model']
                    f1_score = row['F1_Score']
                    details = row['Details']
                    multi_class_f1_scores.append((model_name, method, label, f1_score, details))

# Check if the binary model keys exist and find the best model for each binary method
best_binary_models = {}
for label in binary_f1_scores:
    if binary_f1_scores[label]:
        best_binary_models[label] = max(binary_f1_scores[label], key=lambda x: x[-2])
    else:
        best_binary_models[label] = None

# Find the best model for multi-class classification
best_multi_class_model = max(multi_class_f1_scores, key=lambda x: x[-2], default=None)

# Prepare the data to export into a CSV file
results = {
    'Label Type': ['Binary (Focused)', 'Binary (Unfocused)', 'Binary (Drowsy)', 'Multi-Class'],
    'Method': [
        best_binary_models['binary-focused'][1] if best_binary_models['binary-focused'] else 'N/A',
        best_binary_models['binary-unfocused'][1] if best_binary_models['binary-unfocused'] else 'N/A',
        best_binary_models['binary-drowsy'][1] if best_binary_models['binary-drowsy'] else 'N/A',
        best_multi_class_model[1] if best_multi_class_model else 'N/A'
    ],
    'Best Model': [
        best_binary_models['binary-focused'][0] if best_binary_models['binary-focused'] else 'N/A',
        best_binary_models['binary-unfocused'][0] if best_binary_models['binary-unfocused'] else 'N/A',
        best_binary_models['binary-drowsy'][0] if best_binary_models['binary-drowsy'] else 'N/A',
        best_multi_class_model[0] if best_multi_class_model else 'N/A'
    ],
    'F1_Score': [
        best_binary_models['binary-focused'][-2] if best_binary_models['binary-focused'] else 'N/A',
        best_binary_models['binary-unfocused'][-2] if best_binary_models['binary-unfocused'] else 'N/A',
        best_binary_models['binary-drowsy'][-2] if best_binary_models['binary-drowsy'] else 'N/A',
        best_multi_class_model[-2] if best_multi_class_model else 'N/A'
    ],
    'Details': [
        best_binary_models['binary-focused'][-1] if best_binary_models['binary-focused'] else 'N/A',
        best_binary_models['binary-unfocused'][-1] if best_binary_models['binary-unfocused'] else 'N/A',
        best_binary_models['binary-drowsy'][-1] if best_binary_models['binary-drowsy'] else 'N/A',
        best_multi_class_model[-1] if best_multi_class_model else 'N/A'
    ]
}

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Export the results to a CSV file
results_df.to_csv(folder_path + 'best_model_f1_scores.csv', index=False)

# Print results
print(f"Best Binary-Focused Model: {best_binary_models['binary-focused'][0] if best_binary_models['binary-focused'] else 'N/A'} with F1_Score: {best_binary_models['binary-focused'][-2] if best_binary_models['binary-focused'] else 'N/A'}")
print(f"Best Binary-Unfocused Model: {best_binary_models['binary-unfocused'][0] if best_binary_models['binary-unfocused'] else 'N/A'} with F1_Score: {best_binary_models['binary-unfocused'][-2] if best_binary_models['binary-unfocused'] else 'N/A'}")
print(f"Best Binary-Drowsy Model: {best_binary_models['binary-drowsy'][0] if best_binary_models['binary-drowsy'] else 'N/A'} with F1_Score: {best_binary_models['binary-drowsy'][-2] if best_binary_models['binary-drowsy'] else 'N/A'}")
print(f"Best Multi-Class Model: {best_multi_class_model[0] if best_multi_class_model else 'N/A'} with F1_Score: {best_multi_class_model[-2] if best_multi_class_model else 'N/A'}")