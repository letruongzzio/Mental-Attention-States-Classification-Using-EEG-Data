import pandas as pd
import numpy as np
import json
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from cuml.naive_bayes import GaussianNB
from cuml.decomposition import PCA
from cuml.preprocessing import StandardScaler
from cuml.preprocessing import LabelEncoder
from cuml.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
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
    def __init__(self, TRAIN_PATH):
        self.data = pd.read_csv(TRAIN_PATH)
        self.features = self.data.drop(columns=['state'])
        self.labels = self.data['state']

        self.label_classes = self.labels.unique()
        self.label_encoder = {label: idx for idx, label in enumerate(self.label_classes)}
        self.encoded_labels = self.labels.map(self.label_encoder).values

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features.values)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_features, self.encoded_labels, test_size=0.2, random_state=42, stratify=self.encoded_labels
        )

        self.models = {
            "LDA": LDA(),
            "LogisticRegression": LogisticRegression(),
            "RandomForest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(),
        }

        self.results_binary = {"PCA": {}, "Lasso": {}, "SelectFromModel": {}, "RFE": {}}
        self.results_multiclass = {"PCA": {}, "Lasso": {}, "SelectFromModel": {}, "RFE": {}}
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

    def apply_lasso(self, X_train, X_test, y_train, y_test, result_list):
        print("Applying LassoCV for feature selection...")
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)
        
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        selected_features = np.where(lasso.coef_ != 0)[0]
        
        for model_name, model in self.models.items():
            print(f"  - Training {model_name} on LassoCV-selected features...")
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
                "selected_features_names": [self.features.columns[i] for i in selected_features],
                "alpha": lasso.alpha_,
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
            if not hasattr(model, "coef_") and not hasattr(model, "feature_importances_"):
                print(f"  - Skipping {model_name}: does not support feature importances or coefficients.")
                continue
            
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
                "selected_features_names": [self.features.columns[i] for i in selected_features],
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
                print(f"    - Skipping {model_name}: does not support feature importances or coefficients.")
                continue
            
            for n_features_to_select in np.round(np.arange(0.1, 1, 0.1), 2):
                n_features = int(n_features_to_select * X_train.shape[1])
                print(f"    - Using {n_features} features...")
                rfe = RFE(estimator=model, n_features_to_select=n_features)
                rfe.fit(X_train, y_train)
                selected_features = rfe.get_support(indices=True)
    
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]
    
                # Train the model on the reduced feature set
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
                    "selected_features_names": [self.features.columns[i] for i in selected_features],
                    "n_features_selected": n_features,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

    def export_results_to_csv(self, result_dict, FILE_PATH):
        rows = []
        for method, model_results in result_dict.items():
            for model_name, data in model_results.items():
                for eval_data in data.get("evaluations", []):
                    rows.append({
                        "Method": method,
                        "Model": model_name,
                        "Accuracy": eval_data["accuracy"],
                        "Precision": eval_data["precision"],
                        "Recall": eval_data["recall"],
                        "F1_Score": eval_data["f1_score"],
                        "Details": eval_data.get("selected_features_indices", "N/A")
                    })

        rows.append({
            "Method": "Average",
            "Model": "All",
            "Accuracy": np.mean([r["Accuracy"] for r in rows]),
            "Precision": np.mean([r["Precision"] for r in rows]),
            "Recall": np.mean([r["Recall"] for r in rows]),
            "F1_Score": np.mean([r["F1_Score"] for r in rows]),
            "Details": "N/A"
        })

        rows.append({
            "Method": "Standard Deviation",
            "Model": "All",
            "Accuracy": np.std([r["Accuracy"] for r in rows]),
            "Precision": np.std([r["Precision"] for r in rows]),
            "Recall": np.std([r["Recall"] for r in rows]),
            "F1_Score": np.std([r["F1_Score"] for r in rows]),
            "Details": "N/A"
        })

        df = pd.DataFrame(rows)
        df.to_csv(FILE_PATH, index=False)
        print(f"Results exported to {FILE_PATH}")

    def evaluate_binary(self):
        print("Evaluating binary classification...")
        for i, class_label in enumerate(self.label_classes):
            binary_y_train = (self.y_train == i).astype(int)
            binary_y_test = (self.y_test == i).astype(int)
    
            label_name = f"binary-{class_label}"  # Add label name for identification
    
            print(f"Evaluating for label: {label_name}...")
            
            for method in ["PCA", "Lasso", "SelectFromModel", "RFE"]:
                if method not in self.results_binary:
                    self.results_binary[method] = {}
                if label_name not in self.results_binary[method]:
                    self.results_binary[method][label_name] = []
    
            # Apply feature selection methods and store results
            self.apply_pca(self.X_train, self.X_test, binary_y_train, binary_y_test, self.results_binary["PCA"][label_name])
            self.apply_lasso(self.X_train, self.X_test, binary_y_train, binary_y_test, self.results_binary["Lasso"][label_name])
            self.apply_selectfrommodel(self.X_train, self.X_test, binary_y_train, binary_y_test, self.results_binary["SelectFromModel"][label_name])
            self.apply_rfe(self.X_train, self.X_test, binary_y_train, binary_y_test, self.results_binary["RFE"][label_name])

    def evaluate_multiclass(self):
        print("Evaluating multi-class classification...")
        label_name = "multi-class"  # Common label for multi-class evaluation
    
        # Create nested structure for results_multiclass if not exists
        for method in ["PCA", "Lasso", "SelectFromModel", "RFE"]:
            if method not in self.results_multiclass:
                self.results_multiclass[method] = {}
            if label_name not in self.results_multiclass[method]:
                self.results_multiclass[method][label_name] = []
    
        # Apply feature selection methods and store results
        self.apply_pca(self.X_train, self.X_test, self.y_train, self.y_test, self.results_multiclass["PCA"][label_name])
        self.apply_lasso(self.X_train, self.X_test, self.y_train, self.y_test, self.results_multiclass["Lasso"][label_name])
        self.apply_selectfrommodel(self.X_train, self.X_test, self.y_train, self.y_test, self.results_multiclass["SelectFromModel"][label_name])
        self.apply_rfe(self.X_train, self.X_test, self.y_train, self.y_test, self.results_multiclass["RFE"][label_name])

    def select_best_method(self):
        print("Selecting the best methods and ranking by F1-score...")
        
        def sort_results_by_f1(results):
            ranked = []
            for method, label_results in results.items():
                for label, model_results in label_results.items():
                    for model_name, evaluations in model_results.items():
                        for evaluation in evaluations:
                            entry = {
                                "label": label,  # Include label name
                                "method": method,
                                "model": model_name,
                                "f1_score": evaluation["f1_score"],
                                "details": evaluation
                            }
                            ranked.append(entry)
            # Sort by F1-score in descending order
            ranked.sort(key=lambda x: x["f1_score"], reverse=True)
            return ranked
    
        # Get ranked results for binary and multi-class classification
        ranked_binary = sort_results_by_f1(self.results_binary)
        ranked_multiclass = sort_results_by_f1(self.results_multiclass)
    
        # Select the best method for binary and multi-class tasks
        self.best_methods["Binary"] = ranked_binary[0] if ranked_binary else None
        self.best_methods["MultiClass"] = ranked_multiclass[0] if ranked_multiclass else None
    
        print("Best methods selected.")
    
        # Save ranked results for later use
        self.ranked_results = {
            "Binary": ranked_binary,
            "MultiClass": ranked_multiclass
        }

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
    training_models = TrainingModels(TRAIN_PATH)
    training_models.run(OUTPUT_PATH)