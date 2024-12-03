import pandas as pd
import numpy as np
import json
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
from cuml.naive_bayes import GaussianNB
from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler
from cuml.preprocessing import LabelEncoder
from cuml.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import os

PARENT_DIRNAME = os.path.expanduser("~/PRML-MidTerm-Project/")
TRAIN_PATH = os.path.join(PARENT_DIRNAME, "data", "df_train.csv")
TEST_PATH = os.path.join(PARENT_DIRNAME, "data", "df_test.csv")
OUTPUT_PATH = PARENT_DIRNAME + "model_implementation/machine_learning_models/output"

class TrainingModels:
    def __init__(self, TRAIN_PATH, TEST_PATH):
        self.train_data = pd.read_csv(TRAIN_PATH)
        self.test_data = pd.read_csv(TEST_PATH)

        self.train_features = self.train_data.drop(columns=['state'])
        self.train_labels = self.train_data['state']

        self.test_features = self.test_data.drop(columns=['state'])
        self.test_labels = self.test_data['state']

        self.label_classes = self.train_labels.unique()
        self.label_encoder = {label: idx for idx, label in enumerate(self.label_classes)}
        self.encoded_train_labels = self.train_labels.map(self.label_encoder).values
        self.encoded_test_labels = self.test_labels.map(self.label_encoder).values

        self.scaler = StandardScaler()
        self.scaled_train_features = self.scaler.fit_transform(self.train_features.values)
        self.scaled_test_features = self.scaler.transform(self.test_features.values)

        self.models = {
            "LDA": LDA()
            # "LogisticRegression": LogisticRegression(max_iter=1000),
            # "AdaBoost": AdaBoostClassifier(),
            # "XGBoost": XGBClassifier(),
            # "RandomForest": RandomForestClassifier()
        }

        self.results_binary = {"PCA": {}, "SelectFromModel": {}, "RFE": {}}
        self.results_multiclass = {"PCA": {}, "SelectFromModel": {}, "RFE": {}}
        self.best_methods = {"Binary": {}, "MultiClass": {}}

    def apply_pca(self, X_train, X_test, y_train, y_test, result_list):
        print("Applying PCA for feature selection...")
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
    
        for n_components in np.round(np.arange(0.7, 0.99, 0.05), 2):            
            print(f"  - Applying PCA with n_components={n_components * 100:.2f}%...")
    
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
    
            for model_name, model in self.models.items():
                print(f"    - Training {model_name} on PCA-selected features...")
    
                model.fit(X_train_pca, y_train)
                predictions = model.predict(X_test_pca)
    
                accuracy = round(np.divide(
                    accuracy_score(y_test, predictions),
                    1,
                    where=(accuracy_score(y_test, predictions) != 0),
                ), 4)
                
                precision = round(np.divide(
                    precision_score(y_test, predictions, average='weighted'),
                    1,
                    where=(precision_score(y_test, predictions, average='weighted') != 0),
                ), 4)
                
                recall = round(np.divide(
                    recall_score(y_test, predictions, average='weighted'),
                    1,
                    where=(recall_score(y_test, predictions, average='weighted') != 0),
                ), 4)
                
                f1 = round(np.divide(
                    f1_score(y_test, predictions, average='weighted'),
                    1,
                    where=(f1_score(y_test, predictions, average='weighted') != 0),
                ), 4)
    
                result_list.append({
                    "model": model_name,
                    "n_components (%)": n_components,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

    def apply_selectfrommodel(self, X_train, X_test, y_train, y_test, result_list):
        print("Applying SelectFromModel for feature selection...")
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
    
        for model_name, model in self.models.items():
            print(f"  - Training {model_name} on SelectFromModel-selected features...")

            selector = SelectFromModel(model)
            selector.fit(X_train, y_train)
            selected_features = selector.get_support(indices=True)
    
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            model.fit(X_train_selected, y_train)
            predictions = model.predict(X_test_selected)
            
            accuracy = round(np.divide(
                accuracy_score(y_test, predictions),
                1,
                where=(accuracy_score(y_test, predictions) != 0),
            ), 4)
            
            precision = round(np.divide(
                precision_score(y_test, predictions, average='weighted'),
                1,
                where=(precision_score(y_test, predictions, average='weighted') != 0),
            ), 4)
            
            recall = round(np.divide(
                recall_score(y_test, predictions, average='weighted'),
                1,
                where=(recall_score(y_test, predictions, average='weighted') != 0),
            ), 4)
            
            f1 = round(np.divide(
                f1_score(y_test, predictions, average='weighted'),
                1,
                where=(f1_score(y_test, predictions, average='weighted') != 0),
            ), 4)

            result_list.append({
                "model": model_name,
                "selected_features_indices": selected_features.tolist(),
                "selected_features_names": [self.train_features.columns[i] for i in selected_features],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

    def apply_rfe(self, X_train, X_test, y_train, y_test, result_list):
        print("Applying RFE for feature selection...")
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        for model_name, model in self.models.items():
            print(f"  - Applying RFE with model: {model_name}")
            
            model.fit(X_train, y_train)

            if not hasattr(model, "coef_") and not hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                for n_features_to_select in np.round(np.arange(0.1, 1, 0.1), 2):
                    n_features = int(n_features_to_select * X_train.shape[1])
                    print(f"    - Using top {n_features} features based on importance...")
                    selected_features = indices[:n_features]

                    X_train_selected = X_train[:, selected_features]
                    X_test_selected = X_test[:, selected_features]
                    
                    model.fit(X_train_selected, y_train)
                    predictions = model.predict(X_test_selected)

                    accuracy = round(np.divide(
                        accuracy_score(y_test, predictions),
                        1,
                        where=(accuracy_score(y_test, predictions) != 0),
                    ), 4)
                    
                    precision = round(np.divide(
                        precision_score(y_test, predictions, average='weighted'),
                        1,
                        where=(precision_score(y_test, predictions, average='weighted') != 0),
                    ), 4)
                    
                    recall = round(np.divide(
                        recall_score(y_test, predictions, average='weighted'),
                        1,
                        where=(recall_score(y_test, predictions, average='weighted') != 0),
                    ), 4)
                    
                    f1 = round(np.divide(
                        f1_score(y_test, predictions, average='weighted'),
                        1,
                        where=(f1_score(y_test, predictions, average='weighted') != 0),
                    ), 4)

                    result_list.append({
                        "model": model_name,
                        "selected_features_indices": selected_features.tolist(),
                        "selected_features_names": [self.train_features.columns[i] for i in selected_features],
                        "n_features_selected": n_features,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    })

            else:
                for n_features_to_select in np.round(np.arange(0.1, 1, 0.1), 2):
                    n_features = int(n_features_to_select * X_train.shape[1])
                    print(f"    - Using {n_features} features...")
                    rfe = RFE(estimator=model, n_features_to_select=n_features)
                    rfe.fit(X_train, y_train)
                    selected_features = rfe.get_support(indices=True)

                    X_train_selected = X_train[:, selected_features]
                    X_test_selected = X_test[:, selected_features]

                    model.fit(X_train_selected, y_train)
                    predictions = model.predict(X_test_selected)

                    accuracy = round(np.divide(
                        accuracy_score(y_test, predictions),
                        1,
                        where=(accuracy_score(y_test, predictions) != 0),
                    ), 4)
                    
                    precision = round(np.divide(
                        precision_score(y_test, predictions, average='weighted'),
                        1,
                        where=(precision_score(y_test, predictions, average='weighted') != 0),
                    ), 4)
                    
                    recall = round(np.divide(
                        recall_score(y_test, predictions, average='weighted'),
                        1,
                        where=(recall_score(y_test, predictions, average='weighted') != 0),
                    ), 4)
                    
                    f1 = round(np.divide(
                        f1_score(y_test, predictions, average='weighted'),
                        1,
                        where=(f1_score(y_test, predictions, average='weighted') != 0),
                    ), 4)

                    result_list.append({
                        "model": model_name,
                        "selected_features_indices": selected_features.tolist(),
                        "selected_features_names": [self.train_features.columns[i] for i in selected_features],
                        "n_features_selected": n_features,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    })

    def export_results_to_csv(self, result_dict, FILE_PATH):
        rows = []
    
        for method, model_results in result_dict.items():
            for model_name, evaluations in model_results.items():
                for eval_data in evaluations:
                    rows.append({
                        "Method": method,
                        "Model": model_name,
                        "Accuracy": eval_data.get("accuracy", None),
                        "Precision": eval_data.get("precision", None),
                        "Recall": eval_data.get("recall", None),
                        "F1_Score": eval_data.get("f1_score", None),
                        "Details": eval_data.get("selected_features_indices", "N/A")
                    })
    
        if not rows:
            print("Warning: No results to export. The result dictionary is empty.")
            return
    
        metric_keys = ["Accuracy", "Precision", "Recall", "F1_Score"]
        averages = {metric: np.mean([row[metric] for row in rows if row[metric] is not None]) for metric in metric_keys}
        std_devs = {metric: np.std([row[metric] for row in rows if row[metric] is not None]) for metric in metric_keys}
    
        # Append averages
        rows.append({
            "Method": "Average",
            "Model": "All",
            "Accuracy": round(averages["Accuracy"], 4) if averages["Accuracy"] is not None else None,
            "Precision": round(averages["Precision"], 4) if averages["Precision"] is not None else None,
            "Recall": round(averages["Recall"], 4) if averages["Recall"] is not None else None,
            "F1_Score": round(averages["F1_Score"], 4) if averages["F1_Score"] is not None else None,
            "Details": "N/A"
        })
    
        # Append standard deviations
        rows.append({
            "Method": "Standard Deviation",
            "Model": "All",
            "Accuracy": round(std_devs["Accuracy"], 4) if std_devs["Accuracy"] is not None else None,
            "Precision": round(std_devs["Precision"], 4) if std_devs["Precision"] is not None else None,
            "Recall": round(std_devs["Recall"], 4) if std_devs["Recall"] is not None else None,
            "F1_Score": round(std_devs["F1_Score"], 4) if std_devs["F1_Score"] is not None else None,
            "Details": "N/A"
        })
    
        df = pd.DataFrame(rows)
        df.to_csv(FILE_PATH, index=False)
        print(f"Results exported to {FILE_PATH}")

    def evaluate_binary(self):
        """
        Evaluates binary classification for each class label.
    
        For each class label:
        - Converts the multi-class labels into binary labels (one-vs-rest format).
        - Initializes result storage in the `results_binary` dictionary for the class label.
        - Applies the specified feature selection methods (e.g., PCA, SelectFromModel, RFE).
        - Stores evaluation metrics for each model and feature selection method.
    
        Steps:
        1. Iterate through unique class labels to generate binary classification tasks.
        2. Use the `apply_selectfrommodel` method to evaluate feature selection and model performance.
        3. Dynamically create storage for results if it does not already exist.
        4. Commented-out PCA and RFE for flexibility, but they can be enabled if needed.
    
        Parameters:
        - None (uses internal class attributes such as `scaled_train_features`, `scaled_test_features`,
          and `encoded_train_labels`).
    
        Notes:
        - `binary_y_train` and `binary_y_test` are derived as one-vs-rest binary labels for the current class.
        - `scaled_train_features` and `scaled_test_features` are used as input features.
        - Results are stored in `self.results_binary` under the appropriate method and label name.
    
        Outputs:
        - Evaluation results are appended to `self.results_binary`.
        """
        print("Evaluating binary classification...")
        for i, class_label in enumerate(self.label_classes):
            binary_y_train = (self.encoded_train_labels == i).astype(int)
            binary_y_test = (self.encoded_test_labels == i).astype(int) 
    
            label_name = f"binary-{class_label}"
    
            print(f"Evaluating for label: {label_name}...")
            
            for method in ["PCA", "SelectFromModel", "RFE"]:
                if method not in self.results_binary:
                    self.results_binary[method] = {}
                if label_name not in self.results_binary[method]:
                    self.results_binary[method][label_name] = []

            # PCA
            self.apply_pca(
                self.scaled_train_features, 
                self.scaled_test_features, 
                binary_y_train, binary_y_test, 
                self.results_binary["PCA"][label_name]
            )
    
            # SelectFromModel
            self.apply_selectfrommodel(
                self.scaled_train_features,
                self.scaled_test_features, 
                binary_y_train, 
                binary_y_test, 
                self.results_binary["SelectFromModel"][label_name]
            )
    
            # RFE
            self.apply_rfe(
                self.scaled_train_features, 
                self.scaled_test_features, 
                binary_y_train, binary_y_test, 
                self.results_binary["RFE"][label_name]
            )

    def evaluate_multiclass(self):
        print("Evaluating multi-class classification...")
    
        label_name = "multi-class"
    
        for method in ["PCA", "SelectFromModel", "RFE"]:
            if method not in self.results_multiclass:
                self.results_multiclass[method] = {}
            if label_name not in self.results_multiclass[method]:
                self.results_multiclass[method][label_name] = []
    
        # Apply PCA
        self.apply_pca(
            self.scaled_train_features,
            self.scaled_test_features,
            self.encoded_train_labels,
            self.encoded_test_labels,
            self.results_multiclass["PCA"][label_name]
        )
    
        # Apply SelectFromModel
        self.apply_selectfrommodel(
            self.scaled_train_features,
            self.scaled_test_features,
            self.encoded_train_labels,
            self.encoded_test_labels,
            self.results_multiclass["SelectFromModel"][label_name]
        )
    
        # Apply RFE
        self.apply_rfe(
            self.scaled_train_features,
            self.scaled_test_features,
            self.encoded_train_labels,
            self.encoded_test_labels,
            self.results_multiclass["RFE"][label_name]
        )

    def save_results(self, output_path_base):
        print("Saving results to JSON files...")
        
        # Save best methods
        best_methods_path = f"{output_path_base}_best_methods.json"
        with open(best_methods_path, 'w') as file:
            json.dump({"Best_Methods": self.best_methods}, file, indent=4)
        
        # Save binary classification results
        binary_results_path = f"{output_path_base}_binary_classification.json"
        with open(binary_results_path, 'w') as file:
            json.dump({"Binary_Classification": self.results_binary}, file, indent=4)
        
        # Save multi-class classification results
        multiclass_results_path = f"{output_path_base}_multiclass_classification.json"
        with open(multiclass_results_path, 'w') as file:
            json.dump({"Multi_Class_Classification": self.results_multiclass}, file, indent=4)
        
        print(f"Results saved to:\n- {best_methods_path}\n- {binary_results_path}\n- {multiclass_results_path}")
        print("Saving completed successfully!")

    def run(self, output_path_base="results"):
        print("Starting the feature selection process...")
        
        # Evaluate binary classification
        print("Step 1: Evaluating binary classification...")
        self.evaluate_binary()
        
        # Evaluate multi-class classification
        print("Step 2: Evaluating multi-class classification...")
        self.evaluate_multiclass()
        
        # Export results to CSV
        print("Step 3: Exporting results to CSV...")
        self.export_results_to_csv(self.results_binary, f"{output_path_base}_binary.csv")
        self.export_results_to_csv(self.results_multiclass, f"{output_path_base}_multiclass.csv")
        
        # Save results to JSON
        print("Step 4: Saving results to JSON files...")
        self.save_results(output_path_base)
        
        print("\nFeature selection process completed successfully!")

if __name__ == "__main__":
    training_models = TrainingModels(TRAIN_PATH, TEST_PATH)
    training_models.run(OUTPUT_PATH)
